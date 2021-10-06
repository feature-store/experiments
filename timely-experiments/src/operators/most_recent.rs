use std::{collections::HashMap, hash::Hash};

use timely::{
    dataflow::{operators::Map, Scope},
    Data,
};

use differential_dataflow::{AsCollection, Collection};

/// Maintains only the most recent value for each key.
pub trait MostRecent<S, K, V>
where
    S: Scope,
    K: Data + Hash + Eq,
    V: Data,
{
    fn most_recent(&self) -> Collection<S, (K, V)>;
}

impl<S, K, V> MostRecent<S, K, V> for Collection<S, (K, V)>
where
    S: Scope,
    K: Data + Hash + Eq,
    V: Data,
{
    fn most_recent(&self) -> Collection<S, (K, V)> {
        let mut current_value = HashMap::new();
        self.inner
            .flat_map(move |((key, record), time, diff)| {
                if diff < 0 {
                    unimplemented!("Retractions are unsupported in MostRecent");
                }
                if diff > 1 {
                    unimplemented!("Batched additions are unsupported in MostRecent");
                }
                let mut result = Vec::with_capacity(2);
                if let Some(prev_record) = current_value.remove(&key) {
                    result.push(((key.clone(), prev_record), time.clone(), -1));
                }
                result.push(((key.clone(), record.clone()), time, 1));
                current_value.insert(key, record);

                result
            })
            .as_collection()
    }
}
