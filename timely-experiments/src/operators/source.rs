use std::{
    thread,
    time::{Duration, Instant},
};

use redis::{
    streams::{StreamId, StreamKey, StreamReadOptions, StreamReadReply},
    Commands, Value,
};
use timely::dataflow::{
    operators::{
        generic::{source, OperatorInfo},
        Capability,
    },
    Scope, ScopeParent,
};

use differential_dataflow::{AsCollection, Collection};

use crate::Record;

pub fn fake_source<S: Scope + ScopeParent<Timestamp = usize>>(
    scope: &S,
    name: &str,
    num_keys: usize,
    send_rate_hz: f32,
    num_iters: usize,
) -> Collection<S, (usize, Record<usize, f32>)> {
    source(
        scope,
        name,
        |capability: Capability<S::Timestamp>, info: OperatorInfo| {
            let activator = scope.activator_for(&info.address[..]);
            let mut cap = Some(capability);
            let period = Duration::from_secs_f32(1.0 / send_rate_hz);
            let mut i = 0;
            let mut next_expected_start = None;

            move |output| {
                let start = next_expected_start.unwrap_or_else(Instant::now);
                next_expected_start = Some(start + period);

                let mut done = false;
                if let Some(cap) = cap.as_mut() {
                    // get some data and send it.
                    let time: usize = cap.time().clone();

                    let record = Record::new(i / num_keys, i % num_keys, 1.0);
                    output.session(&cap).give(((record.key, record), time, 1));

                    // downgrade capability.
                    cap.downgrade(&(time + 1));

                    done = time > num_iters;
                }

                if done {
                    cap = None;
                } else {
                    activator.activate();
                    // while Instant::now() < next_expected_start.unwrap() {}

                    let exec_duration = Instant::now() - start;
                    if let Some(sleep_duration) = period.checked_sub(exec_duration) {
                        thread::sleep(sleep_duration);
                    }
                }
                i += 1;
            }
        },
    )
    .as_collection()
}

pub fn redis_source<S: Scope + ScopeParent<Timestamp = usize>>(
    scope: &S,
    name: &str,
    topic: &str,
) -> Collection<S, (usize, Record<usize, f32>)> {
    source(
        scope,
        name,
        |capability: Capability<S::Timestamp>, info: OperatorInfo| {
            let activator = scope.activator_for(&info.address[..]);
            let mut cap = Some(capability);
            let client = redis::Client::open("redis://127.0.0.1/").expect("client");
            let mut con = client.get_connection().expect("con");

            let created: Result<(), _> =
                con.xgroup_create_mkstream(topic, "ralf-reader-group", "$");
            if let Err(e) = created {
                println!("Group already exists: {:?}", e)
            }

            let opts = StreamReadOptions::default()
                .block(1000)
                .count(1)
                .group("ralf-reader-group", "reader-0");

            move |output| {
                let mut done = false;
                if let Some(cap) = cap.as_mut() {
                    // get some data and send it.
                    let time: usize = cap.time().clone();

                    let read_reply: StreamReadReply =
                        con.xread_options(&["ralf"], &[">"], &opts).unwrap();
                    for StreamKey { key: _, ids } in read_reply.keys {
                        for StreamId { id: _, map } in &ids {
                            let timestamp = match map.get("timestamp") {
                                Some(Value::Data(bytes)) => {
                                    // TODO: can remove the clone to be more efficient.
                                    let s = String::from_utf8(bytes.clone()).expect("utf8");
                                    s.parse().unwrap()
                                }
                                _ => panic!("timestamp should be an integer."),
                            };
                            let key = match map.get("key") {
                                Some(Value::Data(bytes)) => {
                                    // TODO: can remove the clone to be more efficient.
                                    let s = String::from_utf8(bytes.clone()).expect("utf8");
                                    s.parse().unwrap()
                                }
                                _ => panic!("key should be an integer."),
                            };
                            let value = match map.get("value") {
                                Some(Value::Data(bytes)) => {
                                    // TODO: can remove the clone to be more efficient.
                                    let s = String::from_utf8(bytes.clone()).expect("utf8");
                                    s.parse().unwrap()
                                }
                                _ => panic!("value should be an integer."),
                            };
                            let send_time: f64 = match map.get("send_time") {
                                Some(Value::Data(bytes)) => {
                                    // TODO: can remove the clone to be more efficient.
                                    let s = String::from_utf8(bytes.clone()).expect("utf8");
                                    s.parse().unwrap()
                                }
                                _ => panic!("send_time should be a string"),
                            };

                            let create_time_ns = (send_time * 1e9) as u128;

                            let record =
                                Record::new_with_create_time(timestamp, key, value, create_time_ns);

                            output.session(&cap).give(((record.key, record), time, 1));
                        }
                    }

                    // downgrade capability.
                    cap.downgrade(&(time + 1));

                    // done = time > num_iters;
                }

                if done {
                    cap = None;
                } else {
                    activator.activate();
                    // while Instant::now() < next_expected_start.unwrap() {}
                }
            }
        },
    )
    .as_collection()
}
