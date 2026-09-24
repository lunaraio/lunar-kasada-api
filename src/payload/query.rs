use url::Url;

pub const V_PARAM: &str = "x-kpsdk-v";
pub const IM_PARAM: &str = "x-kpsdk-im";
pub const CHAMPSSPORTS_PARAM: &str = "ak_bmsc_chmps";
pub const FOOTLOCKER_PARAM: &str = "ak_bmsc_fl_com";
pub const KP_UIDZ_PARAM: &str = "KP_UIDz";
const CHAMPSSPORTS_LABEL: &str = "champssports";
const FOOTLOCKER_LABEL: &str = "footlocker";
const SCHEELS_LABEL: &str = "scheels";
const NIKE_LABEL: &str = "nike";
const COSTCO_LABEL: &str = "costco";
const TICKETMASTER_LABEL: &str = "ticketmaster";
const TWITCH_LABEL: &str = "twitchcdn";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    ChampsSports,
    Footlocker,
    Scheels,
    Nike,
    Costco,
    Ticketmaster,
    Twitch,
}

impl Site {
    pub fn from_host(host: &str) -> Option<Self> {
        let mut labels = host.trim_end_matches('.').rsplit('.');
        labels.next().filter(|tld| !tld.is_empty())?;
        let name = labels.next()?;
        if name.eq_ignore_ascii_case(CHAMPSSPORTS_LABEL) {
            Some(Self::ChampsSports)
        } else if name.eq_ignore_ascii_case(FOOTLOCKER_LABEL) {
            Some(Self::Footlocker)
        } else if name.eq_ignore_ascii_case(SCHEELS_LABEL) {
            Some(Self::Scheels)
        } else if name.eq_ignore_ascii_case(NIKE_LABEL) {
            Some(Self::Nike)
        } else if name.eq_ignore_ascii_case(COSTCO_LABEL) {
            Some(Self::Costco)
        } else if name.eq_ignore_ascii_case(TICKETMASTER_LABEL) {
            Some(Self::Ticketmaster)
        } else if name.eq_ignore_ascii_case(TWITCH_LABEL) {
            Some(Self::Twitch)
        } else {
            None
        }
    }

    pub const fn param_name(self) -> &'static str {
        match self {
            Self::ChampsSports => CHAMPSSPORTS_PARAM,
            Self::Footlocker => FOOTLOCKER_PARAM,
            Self::Scheels | Self::Nike | Self::Costco | Self::Ticketmaster | Self::Twitch => KP_UIDZ_PARAM,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    #[error("invalid ips_link: {0}")]
    InvalidUrl(#[from] url::ParseError),
    #[error("ips_link has no host")]
    MissingHost,
    #[error("unsupported domain {0}")]
    UnsupportedDomain(String),
    #[error("{0} is missing")]
    MissingParam(&'static str),
    #[error("{0} is empty")]
    EmptyParam(&'static str),
}

#[derive(Clone, Copy, Debug)]
pub struct IpsQuery<'a> {
    pub v: &'a str,
    pub im: &'a str,
}

fn require<'a>(slot: Option<&'a str>, name: &'static str) -> Result<&'a str, QueryError> {
    match slot {
        Some("") => Err(QueryError::EmptyParam(name)),
        Some(value) => Ok(value),
        None => Err(QueryError::MissingParam(name)),
    }
}

impl<'a> IpsQuery<'a> {
    pub fn parse(url: &'a Url) -> Result<Self, QueryError> {
        let host = url.host_str().ok_or(QueryError::MissingHost)?;
        let site = Site::from_host(host).ok_or_else(|| QueryError::UnsupportedDomain(host.to_owned()))?;
        let site_param_name = site.param_name();
        let mut v: Option<&'a str> = None;
        let mut im: Option<&'a str> = None;
        let mut site_param_value: Option<&'a str> = None;
        if let Some(query) = url.query() {
            for pair in query.split('&') {
                let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
                if key == V_PARAM {
                    v.get_or_insert(value);
                } else if key == IM_PARAM {
                    im.get_or_insert(value);
                } else if key == site_param_name {
                    site_param_value.get_or_insert(value);
                } else {
                    continue;
                }
                if v.is_some() && im.is_some() && site_param_value.is_some() {
                    break;
                }
            }
        }
        let v = require(v, V_PARAM)?;
        let im = require(im, IM_PARAM)?;
        require(site_param_value, site_param_name)?;
        Ok(Self { v, im })
    }
}
