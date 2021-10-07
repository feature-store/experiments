use timely::dataflow::{operators::Map, Scope};

use differential_dataflow::{AsCollection, Collection};

use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};

use crate::{Record, RecordKey, RecordValue};

pub trait STLFit<S: Scope, K: RecordKey> {
    fn stl_fit(&self, seasonality: usize) -> Collection<S, (K, Record<K, Vec<u8>>)>;
}

impl<S, K, V> STLFit<S, K> for Collection<S, (K, Vec<Record<K, V>>)>
where
    S: Scope,
    K: RecordKey + ToPyObject,
    V: 'static + RecordValue + ToPyObject,
{
    fn stl_fit(&self, seasonality: usize) -> Collection<S, (K, Record<K, Vec<u8>>)> {
        self.map(move |(k, window)| {
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
            (
                k.clone(),
                Record::new_with_create_time(timestamp, k, model, create_time_ns),
            )
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
