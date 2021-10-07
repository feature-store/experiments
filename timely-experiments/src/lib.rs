use std::{
    collections::HashMap,
    fs::File,
    time::{SystemTime, UNIX_EPOCH},
};

pub mod operators;
mod record;

pub use record::{Record, RecordKey, RecordValue};

pub fn ns_since_unix_epoch() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

pub fn parse_per_key_slide_size(filename: &str) -> HashMap<usize, usize> {
    let file = File::open(filename).unwrap();
    serde_json::from_reader(file).unwrap()
}
