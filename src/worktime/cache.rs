use std::borrow::Cow;
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Instant;

use rustc_hash::FxHashMap;
use tokio::sync::watch;
use wreq::Client;

use super::fetch::{self, FetchError, Lifetime};
use super::pipeline::Extracted;

const KEY_STACK: usize = 256;
const ABORTED: &str = "p.js fetch aborted";
pub type Outcome = Result<Arc<Extracted>, FetchError>;
type Receiver = watch::Receiver<Option<Outcome>>;
type Sender = watch::Sender<Option<Outcome>>;
#[derive(Clone)]
pub struct Cache {
    inner: Arc<RwLock<Table>>,
}

struct Table {
    slots: FxHashMap<Box<str>, Slot>,
    next_generation: u64,
}

struct Slot {
    entry: Option<Entry>,
    flight: Option<Flight>,
}

struct Entry {
    value: Arc<Extracted>,
    fresh_until: Instant,
    stale_until: Instant,
    generation: u64,
}

struct Flight {
    generation: u64,
    rx: Receiver,
}

enum Plan {
    Serve(Arc<Extracted>),
    Join(Receiver),
    Refresh(Arc<Extracted>),
    Fetch,
}

enum Claim {
    Serve(Arc<Extracted>),
    Join(Receiver),
    Refresh(Arc<Extracted>, u64, Sender),
    Fetch(Receiver, u64, Sender),
}

enum Settled<'a> {
    Cached(&'a Arc<Extracted>, Lifetime),
    Uncached,
    Failed,
}

struct Abandon<'a> {
    cache: &'a Cache,
    key: &'a str,
    generation: u64,
    armed: bool,
}

impl Drop for Abandon<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.cache.settle(self.key, self.generation, Settled::Failed);
        }
    }
}

impl Cache {
    pub fn new() -> Cache {
        Cache {
            inner: Arc::new(RwLock::new(Table {
                slots: FxHashMap::default(),
                next_generation: 0,
            })),
        }
    }

    pub async fn resolve(&self, client: &Client, domain: &str) -> Outcome {
        let mut buf = [0u8; KEY_STACK];
        let key = lower(domain, &mut buf);
        let now = Instant::now();
        let planned = {
            let table = self.read();
            match table.slots.get(&*key) {
                Some(slot) => plan(slot, now),
                None => Plan::Fetch,
            }
        };
        let rx = match planned {
            Plan::Serve(value) => return Ok(value),
            Plan::Join(rx) => rx,
            Plan::Refresh(_) | Plan::Fetch => match self.claim(&key, now) {
                Claim::Serve(value) => return Ok(value),
                Claim::Join(rx) => rx,
                Claim::Refresh(value, generation, tx) => {
                    self.launch(client, &key, domain, generation, tx);
                    return Ok(value);
                }
                Claim::Fetch(rx, generation, tx) => {
                    self.launch(client, &key, domain, generation, tx);
                    rx
                }
            },
        };
        wait(rx).await
    }

    fn read(&self) -> RwLockReadGuard<'_, Table> {
        self.inner.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, Table> {
        self.inner.write().unwrap_or_else(PoisonError::into_inner)
    }

    fn claim(&self, key: &str, now: Instant) -> Claim {
        let mut guard = self.write();
        let table = &mut *guard;
        let generation = table.next_generation;
        let slot = table.slots.entry(Box::from(key)).or_insert(Slot {
            entry: None,
            flight: None,
        });
        match plan(slot, now) {
            Plan::Serve(value) => Claim::Serve(value),
            Plan::Join(rx) => Claim::Join(rx),
            Plan::Refresh(value) => {
                let (_, tx) = open(slot, generation, &mut table.next_generation);
                Claim::Refresh(value, generation, tx)
            }
            Plan::Fetch => {
                let (rx, tx) = open(slot, generation, &mut table.next_generation);
                Claim::Fetch(rx, generation, tx)
            }
        }
    }

    fn launch(&self, client: &Client, key: &str, domain: &str, generation: u64, tx: Sender) {
        tokio::spawn(run(self.clone(), client.clone(), Box::from(key), Box::from(domain), generation, tx));
    }

    fn settle(&self, key: &str, generation: u64, settled: Settled<'_>) {
        let mut guard = self.write();
        let table = &mut *guard;
        let Some(slot) = table.slots.get_mut(key) else {
            return;
        };
        if slot.flight.as_ref().is_some_and(|f| f.generation == generation) {
            slot.flight = None;
        }
        let newer = slot.entry.as_ref().is_some_and(|e| e.generation > generation);
        match settled {
            Settled::Cached(value, lifetime) => {
                if !newer {
                    slot.entry = Some(Entry {
                        value: value.clone(),
                        fresh_until: lifetime.fresh_until,
                        stale_until: lifetime.stale_until,
                        generation,
                    });
                }
            }
            Settled::Uncached => {
                if !newer {
                    slot.entry = None;
                }
            }
            Settled::Failed => {
                let now = Instant::now();
                if slot.entry.as_ref().is_some_and(|e| now >= e.stale_until) {
                    slot.entry = None;
                }
            }
        }
        if slot.entry.is_none() && slot.flight.is_none() {
            table.slots.remove(key);
        }
    }
}

fn lower<'a>(domain: &'a str, buf: &'a mut [u8; KEY_STACK]) -> Cow<'a, str> {
    let n = domain.len();
    if n > KEY_STACK {
        return Cow::Owned(domain.to_ascii_lowercase());
    }
    let dst = &mut buf[..n];
    dst.copy_from_slice(domain.as_bytes());
    dst.make_ascii_lowercase();
    match std::str::from_utf8(dst) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(domain.to_ascii_lowercase()),
    }
}

fn plan(slot: &Slot, now: Instant) -> Plan {
    if let Some(entry) = &slot.entry {
        if now < entry.fresh_until {
            return Plan::Serve(entry.value.clone());
        }
        if now < entry.stale_until {
            return match slot.flight {
                Some(_) => Plan::Serve(entry.value.clone()),
                None => Plan::Refresh(entry.value.clone()),
            };
        }
    }
    match &slot.flight {
        Some(flight) => Plan::Join(flight.rx.clone()),
        None => Plan::Fetch,
    }
}

fn open(slot: &mut Slot, generation: u64, next_generation: &mut u64) -> (Receiver, Sender) {
    let (tx, rx) = watch::channel(None);
    *next_generation = generation + 1;
    slot.flight = Some(Flight {
        generation,
        rx: rx.clone(),
    });
    (rx, tx)
}

async fn run(cache: Cache, client: Client, key: Box<str>, domain: Box<str>, generation: u64, tx: Sender) {
    let mut guard = Abandon {
        cache: &cache,
        key: &key,
        generation,
        armed: true,
    };
    let outcome = match fetch::fetch(&client, &domain).await {
        Ok(fetched) => {
            let value = Arc::new(fetched.extracted);
            match fetched.lifetime {
                Some(lifetime) => cache.settle(&key, generation, Settled::Cached(&value, lifetime)),
                None => cache.settle(&key, generation, Settled::Uncached),
            }
            Ok(value)
        }
        Err(e) => {
            cache.settle(&key, generation, Settled::Failed);
            Err(e)
        }
    };
    guard.armed = false;
    drop(guard);
    tx.send_replace(Some(outcome));
}

async fn wait(mut rx: Receiver) -> Outcome {
    match rx.wait_for(Option::is_some).await {
        Ok(current) => match &*current {
            Some(outcome) => outcome.clone(),
            None => Err(FetchError::bad_gateway(ABORTED)),
        },
        Err(_) => Err(FetchError::bad_gateway(ABORTED)),
    }
}
