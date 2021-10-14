use std::{fmt::Debug, hash::Hash};

use abomonation::Abomonation;

use differential_dataflow::Data;

use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};

pub trait RecordKey: Data + Clone + Abomonation + Hash + Send {}
impl<T: Data + Clone + Abomonation + Hash + Send> RecordKey for T {}

pub trait RecordValue: PartialEq + Clone + Abomonation + Send {}
impl<T: PartialEq + Clone + Abomonation + Send> RecordValue for T {}

#[derive(Clone, abomonation_derive::Abomonation, Hash)]
pub struct Record<K: RecordKey, V: RecordValue> {
    pub key: K,
    pub timestamp: usize,
    pub value: V,
    pub create_time_ns: u128,
}

impl<K: RecordKey, V: RecordValue> Record<K, V> {
    pub fn new(timestamp: usize, key: K, value: V) -> Self {
        Self::new_with_create_time(timestamp, key, value, crate::ns_since_unix_epoch())
    }

    pub fn new_with_create_time(timestamp: usize, key: K, value: V, create_time_ns: u128) -> Self {
        Self {
            timestamp,
            key,
            value,
            create_time_ns,
        }
    }
}

impl<K: RecordKey, V: RecordValue> Ord for Record<K, V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.timestamp.cmp(&other.timestamp)
    }
}

impl<K: RecordKey, V: RecordValue> PartialOrd for Record<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: RecordKey, V: RecordValue> PartialEq for Record<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.timestamp == other.timestamp && self.value == other.value
    }
}

impl<K: RecordKey, V: RecordValue> Eq for Record<K, V> {
    fn assert_receiver_is_total_eq(&self) {}
}

impl<K: RecordKey, V: RecordValue + Debug> Debug for Record<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Record @ {}: {{ {:?}: {:?} }}",
            self.timestamp, self.key, self.value
        )
    }
}

impl<K: RecordKey + ToPyObject, V: RecordValue + ToPyObject> IntoPyDict for Record<K, V> {
    fn into_py_dict(self, py: Python) -> &PyDict {
        let dict = PyDict::new(py);
        dict.set_item("key", self.key).unwrap();
        dict.set_item("value", self.value).unwrap();
        dict.set_item("timestamp", self.timestamp).unwrap();
        dict
    }
}
