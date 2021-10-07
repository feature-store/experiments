use std::{collections::HashMap, hash::Hash};

use timely::{
    dataflow::{
        operators::{Exchange, Map},
        Scope,
    },
    Data, ExchangeData,
};

use differential_dataflow::{AsCollection, Collection, Hashable};

pub trait Window<S: Scope, K: Data + ExchangeData, V: Data + ExchangeData> {
    fn sliding_window(&self, window_size: usize, slide_size: usize) -> Collection<S, (K, Vec<V>)>;
    fn variable_sliding_window(
        &self,
        window_size: usize,
        slide_sizes: HashMap<K, usize>,
    ) -> Collection<S, (K, Vec<V>)>;
}

impl<S: Scope, K: Data + Hash + Eq + std::fmt::Debug + ExchangeData, V: Data + ExchangeData>
    Window<S, K, V> for Collection<S, (K, V)>
{
    fn sliding_window(&self, window_size: usize, slide_size: usize) -> Collection<S, (K, Vec<V>)> {
        let mut windows: HashMap<K, Vec<V>> = HashMap::new();
        self.inner
            .exchange(|((k, _), _, _)| k.hashed()) // Assign each key to a worker.
            .flat_map(move |((key, value), time, diff)| {
                if diff <= 0 {
                    unimplemented!("Retractions are unsupported for windowing.")
                }

                let entry = windows
                    .entry(key.clone())
                    .or_insert(Vec::with_capacity(window_size));
                for _ in 1..diff {
                    entry.push(value.clone());
                }
                entry.push(value);

                // TODO: handle negative differences.
                // TODO: can optimize by by only removing if slide_size == window_size.
                // println!(
                //     "time: {:?}, key: {:?}, window_size: {}, thread ID: {:?}",
                //     time,
                //     key,
                //     entry.len(),
                //     std::thread::current().id(),
                // );

                if entry.len() == window_size {
                    let window = windows.remove(&key).unwrap();
                    windows.insert(key.clone(), window[slide_size..].into());
                    vec![((key, window), time, 1)]
                } else {
                    vec![]
                }
            })
            .as_collection()
    }

    fn variable_sliding_window(
        &self,
        window_size: usize,
        slide_sizes: HashMap<K, usize>,
    ) -> Collection<S, (K, Vec<V>)> {
        // TODO: don't duplicate code and use a helper function?
        let mut windows: HashMap<K, Vec<V>> = HashMap::new();
        self.inner
            .exchange(|((k, _), _, _)| k.hashed()) // Assign each key to a worker.
            .flat_map(move |((key, value), time, diff)| {
                if diff <= 0 {
                    unimplemented!("Retractions are unsupported for windowing.")
                }

                let entry = windows
                    .entry(key.clone())
                    .or_insert(Vec::with_capacity(window_size));
                for _ in 1..diff {
                    entry.push(value.clone());
                }
                entry.push(value);

                // TODO: handle negative differences.
                // TODO: can optimize by by only removing if slide_size == window_size.

                if entry.len() == window_size {
                    let window = windows.remove(&key).unwrap();
                    let slide_size: usize = *slide_sizes.get(&key).unwrap();
                    windows.insert(key.clone(), window[slide_size..].into());
                    vec![((key, window), time, 1)]
                } else {
                    vec![]
                }
            })
            .as_collection()
    }
}
