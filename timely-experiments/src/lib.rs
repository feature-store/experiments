use std::time::{SystemTime, UNIX_EPOCH};

pub mod operators;
mod record;

pub use record::{Record, RecordKey, RecordValue};

pub fn ns_since_unix_epoch() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}
