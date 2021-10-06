use redis::Commands;

use timely::dataflow::Scope;

use differential_dataflow::Collection;

use crate::Record;

/// Specialized for models.
/// TODO: generalize this trait.
pub trait ToRedis<S: Scope> {
    /// Stores records in Redis.
    fn to_redis(&self);
}

impl<S: Scope> ToRedis<S> for Collection<S, (usize, Record<usize, Vec<u8>>)> {
    fn to_redis(&self) {
        // TODO: parametrize the connection.
        let client = redis::Client::open("redis://127.0.0.1/2").expect("client");
        let mut con = client.get_connection().expect("con");
        self.inspect(move |((k, record), _t, count)| {
            if *count != 1 {
                todo!("counts != are not supported for Redis");
            }

            // TODO: try to remove extra copy.
            let _: () = con
                .set(format!("{}/models/value", record.key), record.value.clone())
                .unwrap();

            let send_time: f64 = record.create_time_ns as f64 / 1e9;
            let _: () = con
                .set(format!("{}/models/send_time", record.key), send_time)
                .unwrap();

            let insert_time = crate::ns_since_unix_epoch() as f64 / 1e9;
            let _: () = con
                .set(format!("{}/models/create_time", k), insert_time)
                .unwrap();

            let _: () = con
                .set(format!("{}/models/timestamp", k), record.timestamp)
                .unwrap();
        });
    }
}
