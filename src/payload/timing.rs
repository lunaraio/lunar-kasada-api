use rand::RngExt;

const TICK_US: u64 = 100;
const UPTIME_US: (u64, u64) = (1_000_000_000, 100_000_000_000);
const MARKER: u8 = b'1';
const ALPHABET: &[u8; 32] = b"0123456789abcdefghijklmnopqrstuv";
const DELIMS: &[u8; 4] = b"wxyz";
const FIELDS: usize = 14;
const HEADER_CAP: usize = 64;

const NAV_QUEUE_MS: (f64, f64, f64) = (0.6, 1.0, 2.4);
const NAV_TTFB_MS: (f64, f64, f64) = (45.0, 110.0, 260.0);
const NAV_BODY_MS: (f64, f64, f64) = (0.3, 0.9, 2.0);
const IPS_QUEUE_MS: (f64, f64, f64) = (0.6, 1.4, 5.4);
const IPS_TTFB_MS: (f64, f64, f64) = (160.0, 290.0, 530.0);
const IPS_BODY_MS: (f64, f64, f64) = (10.0, 90.0, 420.0);
const IPS_LEAD_MS: (f64, f64, f64) = (0.0, 0.8, 3.5);
const INLINE_MS: (f64, f64, f64) = (0.8, 1.6, 4.0);
const COMPILE_MS: (f64, f64, f64) = (0.2, 0.8, 2.6);
const BOOT_MS: (f64, f64, f64) = (10.0, 18.0, 45.0);
const SETUP_MS: (f64, f64, f64) = (1.6, 3.0, 6.4);
const COLLECT_MS: (f64, f64, f64) = (220.0, 320.0, 900.0);
const LITE_COLLECT_MS: (f64, f64, f64) = (0.1, 0.2, 0.5);
const MIN_COLLECT_FIELD: f64 = 1.0;
const HANDOFF_MS: (f64, f64, f64) = (28.0, 36.0, 50.0);
const ASSEMBLE_MS: (f64, f64, f64) = (88.0, 150.0, 470.0);
const SETTLE_MS: (f64, f64, f64) = (0.1, 0.6, 1.4);
const SEAL_GAP_MS: (f64, f64, f64) = (0.0, 0.3, 1.8);
const ENCRYPT_MS: (f64, f64, f64) = (2.6, 4.2, 6.6);
const RATIO: (f64, f64, f64) = (0.48, 0.51, 0.53);
const HEADER_BYTES: (f64, f64, f64) = (280.0, 420.0, 700.0);

pub struct Transfer {
    pub encoded: Option<u64>,
    pub decoded: usize,
    pub header_bytes: Option<usize>,
}

pub struct Fetch {
    pub start_ms: f64,
    pub ttfb_ms: f64,
    pub body_ms: f64,
}

#[derive(Clone)]
pub struct Timeline {
    uptime_us: u64,
    nav: [u64; 4],
    ips: [u64; 4],
    kjm: u64,
    rbp: u64,
    jlm: u64,
    qbq: u64,
    uxn: u64,
    rcj: u64,
    settle: u64,
    kkm: u64,
    rqk: u64,
    jbp: u64,
    transfer: f64,
}

fn tri(rng: &mut impl RngExt, (lo, mode, hi): (f64, f64, f64)) -> f64 {
    let u: f64 = rng.random();
    let c = (mode - lo) / (hi - lo);
    if u < c { lo + (u * (hi - lo) * (mode - lo)).sqrt() } else { hi - ((1.0 - u) * (hi - lo) * (hi - mode)).sqrt() }
}

fn ticks(ms: f64) -> u64 {
    ((ms.max(0.0) * 1000.0 / TICK_US as f64).round() as u64) * TICK_US
}

fn js_round(x: f64) -> f64 {
    (x + 0.5).floor()
}

fn push_radix32(out: &mut Vec<u8>, n: i64) {
    if n == 0 {
        out.push(b'0');
        return;
    }
    if n < 0 {
        out.push(b'-');
    }
    let mut v = n.unsigned_abs();
    let mut buf = [0u8; 16];
    let mut i = buf.len();
    while v > 0 {
        i -= 1;
        buf[i] = ALPHABET[(v % 32) as usize];
        v /= 32;
    }
    out.extend_from_slice(&buf[i..]);
}

impl Timeline {
    fn build(rng: &mut impl RngExt, nav: [u64; 4], ips: [u64; 4], transfer: f64) -> Self {
        let kjm = nav[3] + ticks(tri(rng, INLINE_MS));
        let rbp = ips[3].max(kjm) + ticks(tri(rng, COMPILE_MS));
        let jlm = rbp + ticks(tri(rng, BOOT_MS));
        let qbq = jlm + ticks(tri(rng, SETUP_MS));
        let uxn = qbq + ticks(tri(rng, COLLECT_MS));
        let rcj = uxn + ticks(tri(rng, HANDOFF_MS));
        let settle = rcj + ticks(tri(rng, SETTLE_MS));
        let kkm = rcj + ticks(tri(rng, ASSEMBLE_MS)).max(settle - rcj + TICK_US);
        let rqk = kkm + ticks(tri(rng, SEAL_GAP_MS));
        let jbp = rqk + ticks(tri(rng, ENCRYPT_MS));
        Timeline {
            uptime_us: rng.random_range(UPTIME_US.0..UPTIME_US.1) / TICK_US * TICK_US,
            nav,
            ips,
            kjm,
            rbp,
            jlm,
            qbq,
            uxn,
            rcj,
            settle,
            kkm,
            rqk,
            jbp,
            transfer,
        }
    }

    fn transfer(rng: &mut impl RngExt, size: &Transfer) -> f64 {
        let encoded = match size.encoded {
            Some(n) => n as f64,
            None => (size.decoded as f64 * tri(rng, RATIO)).round(),
        };
        let headers = match size.header_bytes {
            Some(n) => n as f64,
            None => tri(rng, HEADER_BYTES).round(),
        };
        encoded + headers
    }

    pub fn measured(rng: &mut impl RngExt, nav: &Fetch, ips: &Fetch, size: &Transfer) -> Self {
        let nav_req = ticks(tri(rng, NAV_QUEUE_MS));
        let nav_resp = nav_req + ticks(nav.ttfb_ms);
        let nav_end = nav_resp + ticks(nav.body_ms).max(TICK_US);
        let ips_fetch = ticks(ips.start_ms).max(nav_end);
        let ips_req = ips_fetch + ticks(tri(rng, IPS_QUEUE_MS));
        let ips_resp = ips_req + ticks(ips.ttfb_ms);
        let ips_end = ips_resp + ticks(ips.body_ms).max(TICK_US);
        let transfer = Self::transfer(rng, size);
        Self::build(rng, [0, nav_req, nav_resp, nav_end], [ips_fetch, ips_req, ips_resp, ips_end], transfer)
    }

    pub fn synthetic(rng: &mut impl RngExt, size: &Transfer) -> Self {
        let nav_req = ticks(tri(rng, NAV_QUEUE_MS));
        let nav_resp = nav_req + ticks(tri(rng, NAV_TTFB_MS));
        let nav_end = nav_resp + ticks(tri(rng, NAV_BODY_MS)).max(TICK_US);
        let ips_fetch = nav_end.saturating_sub(ticks(tri(rng, IPS_LEAD_MS))).max(nav_resp);
        let ips_req = ips_fetch + ticks(tri(rng, IPS_QUEUE_MS));
        let ips_resp = ips_req + ticks(tri(rng, IPS_TTFB_MS));
        let ips_end = ips_resp + ticks(tri(rng, IPS_BODY_MS));
        let transfer = Self::transfer(rng, size);
        Self::build(rng, [0, nav_req, nav_resp, nav_end], [ips_fetch, ips_req, ips_resp, ips_end], transfer)
    }

    fn span(&self, from_us: u64, to_us: u64) -> f64 {
        (self.uptime_us + to_us) as f64 / 1000.0 - (self.uptime_us + from_us) as f64 / 1000.0
    }

    pub fn collect(&self) -> f64 {
        self.span(self.qbq, self.uxn)
    }

    pub fn elapsed(&self) -> f64 {
        js_round(self.span(self.settle, self.kkm) * 10.0) / 10.0
    }

    pub fn transfer_rate(&self) -> f64 {
        js_round(self.transfer / self.span(self.ips[0], self.ips[3]))
    }

    pub fn lite(&self, rng: &mut impl RngExt) -> Self {
        let mut t = self.clone();
        let uxn = t.qbq + ticks(tri(rng, LITE_COLLECT_MS)).max(TICK_US);
        let delta = t.uxn - uxn;
        t.uxn = uxn;
        t.rcj -= delta;
        t.settle -= delta;
        t.kkm -= delta;
        t.rqk -= delta;
        t.jbp -= delta;
        t
    }

    pub fn header(&self, rng: &mut impl RngExt) -> String {
        let mut fields: Vec<(u8, f64)> = Vec::with_capacity(FIELDS);
        fields.push((0, self.span(self.kjm, self.jbp) + self.span(self.nav[0], self.nav[3])));
        fields.push((1, self.span(self.kjm, self.rbp)));
        fields.push((2, self.span(self.rbp, self.jlm)));
        fields.push((3, self.span(self.jlm, self.qbq)));
        fields.push((4, self.span(self.qbq, self.uxn).max(MIN_COLLECT_FIELD)));
        fields.push((5, self.span(self.uxn, self.rcj)));
        fields.push((6, self.span(self.rcj, self.kkm)));
        fields.push((7, self.span(self.rqk, self.jbp)));
        for (base, t) in [(8u8, &self.nav), (11u8, &self.ips)] {
            fields.push((base, self.span(t[0], t[1])));
            fields.push((base + 1, self.span(t[1], t[2])));
            fields.push((base + 2, self.span(t[2], t[3])));
        }
        let mut out = Vec::with_capacity(HEADER_CAP);
        out.push(MARKER);
        let mut first = true;
        while !fields.is_empty() {
            let pick = (rng.random::<f64>() * fields.len() as f64).floor() as usize;
            let (id, v) = fields.swap_remove(pick.min(fields.len() - 1));
            let v = js_round(v);
            if !v.is_finite() {
                continue;
            }
            if first {
                first = false;
            } else {
                out.push(DELIMS[(rng.random::<f64>() * DELIMS.len() as f64).floor() as usize]);
            }
            out.push(ALPHABET[id as usize]);
            push_radix32(&mut out, v as i64);
        }
        String::from_utf8(out).unwrap_or_default()
    }
}
