use std::{
    collections::{BinaryHeap, HashMap},
    sync::mpsc,
    thread,
};

use timely::dataflow::{operators::Map, Scope};

use differential_dataflow::{AsCollection, Collection};

use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};

use crate::{Record, RecordKey, RecordValue};

fn run_stl<K, V>(seasonality: usize, window: Vec<Record<K, V>>) -> Record<K, Vec<u8>>
where
    K: RecordKey + ToPyObject,
    V: 'static + RecordValue + ToPyObject,
{
    let key = window[0].key.clone();
    let timestamp = window.last().unwrap().timestamp;
    let create_time_ns = window.last().unwrap().create_time_ns;
    let model: Vec<u8> = Python::with_gil(move |py| {
        let dicts: Vec<&PyDict> = window.into_iter().map(|r| r.into_py_dict(py)).collect();

        py.run(
            "import sys; import os; sys.path.append(os.getcwd() + '/python/'); import stl",
            None,
            None,
        )
        .unwrap();

        let locals = [("window", dicts)].into_py_dict(py);
        locals.set_item("seasonality", seasonality).unwrap();
        py.eval("stl.fit_window(window, seasonality)", None, Some(locals))
            .unwrap()
            .extract()
            .unwrap()
    });
    Record::new_with_create_time(timestamp, key, model, create_time_ns)
}

pub trait STLFit<S: Scope, K: RecordKey> {
    fn stl_fit(&self, seasonality: usize) -> Collection<S, (K, Record<K, Vec<u8>>)>;
    fn stl_fit_lifo(&self, seasonality: usize) -> Collection<S, (K, Record<K, Vec<u8>>)>;
}

impl<S, K, V> STLFit<S, K> for Collection<S, (K, Vec<Record<K, V>>)>
where
    S: Scope,
    K: RecordKey + ToPyObject,
    V: 'static + RecordValue + ToPyObject,
{
    fn stl_fit(&self, seasonality: usize) -> Collection<S, (K, Record<K, Vec<u8>>)> {
        self.map(move |(k, window)| (k, run_stl(seasonality, window)))
    }

    // TODO: find a better way to do LIFO than offloading work in a thread,
    // and sending once the thread has completed and a new window arrives.
    fn stl_fit_lifo(&self, seasonality: usize) -> Collection<S, (K, Record<K, Vec<u8>>)> {
        // A hashmap of windows in LIFO ordering.
        let mut queues: HashMap<K, BinaryHeap<(usize, Vec<Record<K, V>>)>> = HashMap::new();
        // Spawn worker thread which pulls new records to compute and returns values.
        let (operator_tx, worker_rx) = mpsc::channel();
        let (worker_tx, operator_rx) = mpsc::channel();
        thread::Builder::new()
            .name("python-worker".to_string())
            .spawn(move || {
                while let Ok(window) = worker_rx.recv() {
                    let model = run_stl(seasonality, window);
                    if let Err(e) = worker_tx.send(model) {
                        eprintln!("python-worker errored with {:?}", e);
                        return;
                    }
                }
            })
            .unwrap();

        // Used for round-robin processing of keys.
        let mut iter = 0;
        let mut num_items = 0;

        self.flat_map(move |(k, window)| {
            num_items += 1;
            println!("total items: {}", num_items);
            if num_items == 1 {
                operator_tx.send(window).unwrap();
                return vec![];
            }

            // Add to queue.
            let entry = queues.entry(k).or_default();
            entry.push((window[0].timestamp, window));

            if let Ok(record) = operator_rx.try_recv() {
                // Spawn the next task.
                let mut keys: Vec<_> = queues.keys().cloned().collect();
                keys.sort();

                // Choose the next heap containing a window.
                let mut heap_option = None;
                for _ in 0..keys.len() {
                    iter = (iter + 1) % keys.len();
                    heap_option = queues.get_mut(&keys[iter]);
                    if let Some(heap) = heap_option.as_ref() {
                        if !heap.is_empty() {
                            break;
                        }
                    }
                }
                if let Some(heap) = heap_option {
                    // Get the window.
                    let (_t, window_to_process) = heap.pop().unwrap();

                    // Remove all older windows.
                    heap.clear();

                    // Spawn a new thread to process a window.
                    operator_tx.send(window_to_process).unwrap();
                }

                let key = record.key.clone();
                vec![(key, record)]
            } else {
                vec![]
            }
        })
    }
}

pub trait STLInference<S: Scope, K: RecordKey> {
    fn stl_predict(&self) -> Collection<S, (K, Record<K, Vec<u8>>)>;
}

impl<S, K, V> STLInference<S, K> for Collection<S, (K, (Record<K, V>, Record<K, Vec<u8>>))>
where
    S: Scope,
    K: RecordKey + ToPyObject,
    V: 'static + RecordValue + ToPyObject,
{
    fn stl_predict(&self) -> Collection<S, (K, Record<K, Vec<u8>>)> {
        self.inner
        .flat_map(|((key, (point, model)), time, diff)| {
            let timestamp = point.timestamp;
            let create_time_ns = point.create_time_ns.min(model.create_time_ns);
            if diff > 0 {
                let prediction: Vec<u8> = Python::with_gil(move |py| {
                    py.run(
                        "import sys; import os; sys.path.append(os.getcwd() + '/python/'); import stl",
                        None,
                        None,
                    )
                    .unwrap();

                    let locals = [("model", model.into_py_dict(py)), ("point", point.into_py_dict(py))].into_py_dict(py);
                    py.eval("stl.predict(model, point)", None, Some(locals))
                        .unwrap()
                        .extract()
                        .unwrap()
                });

                return vec![((key.clone(), Record::new_with_create_time(timestamp, key,  prediction, create_time_ns)), time, diff)]
            }
            vec![]
        }).as_collection()
    }
}
