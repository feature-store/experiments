extern crate abomonation;
extern crate abomonation_derive;
extern crate clap;
extern crate differential_dataflow;
extern crate timely;
extern crate timely_experiments;

use std::collections::HashMap;

use clap::{App, Arg};

use timely_experiments::{
    operators::{fake_source, redis_source, MostRecent, STLFit, STLInference, ToRedis, Window},
    Record,
};

/// Runs an experiment.
/// source: "redis", or "fake"
/// num_keys, timesteps, send_rate_hz are used if source == "fake"
fn run_experiment(
    source_str: &str,
    global_window_size: usize,
    global_slide_size: usize,
    per_key_slide_size: Option<HashMap<usize, usize>>,
    seasonality: usize,
    prioritization: &str,
    num_keys: usize,
    timesteps: usize,
    send_rate_hz: f32,
    threads: usize,
    process_index: usize,
    num_processes: usize,
) {
    let source_str = source_str.to_string();
    let prioritization_str = prioritization.to_string();

    let config = if num_processes == 1 {
        timely::Config::process(threads)
    } else {
        let mut addresses = vec![];
        for index in 0..num_processes {
            addresses.push(format!("localhost:{}", 2101 + index));
        }

        timely::Config {
            communication: timely::CommunicationConfig::Cluster {
                threads,
                process: process_index,
                addresses,
                report: false,
                log_fn: Box::new(|_| None),
            },
            worker: timely::WorkerConfig::default(),
        }
    };

    timely::execute(config, move |worker| {
        // TODO: look into using Timely Exchange operator to parition across worker processes.
        worker.dataflow(|scope| {
            let source = match source_str.as_str() {
                "redis" => redis_source(scope, "Redis Source", "ralf"),
                "fake" => fake_source(scope, "Fake Source", num_keys, send_rate_hz, timesteps),
                _ => panic!("Invalid source specified."),
            };
            let windows = if let Some(slide_sizes) = per_key_slide_size.as_ref() {
                source.variable_sliding_window(global_window_size, slide_sizes.clone())
            } else {
                source.sliding_window(global_window_size, global_slide_size)
            };
            let models = match prioritization_str.as_str() {
                "fifo" => windows.stl_fit(seasonality),
                "lifo" => windows.stl_fit_lifo(seasonality),
                _ => panic!("Unsupported prioritization strategy."),
            };
            models.to_redis();

            // models.inspect(|((k, v), t, count)| {
            //     println!("model: {}, adjustment: {}", k, count);
            // });

            // let most_recent_models = models.most_recent();
            // most_recent_models
            //     .map(|(k, r)| r.key)
            //     .inspect(|x| println!("{:?}", x));
            // let most_recent_points: Collection<_, _> = source.most_recent();
            // // .consolidate();

            // let to_predict = most_recent_points.join(&most_recent_models);
            // let _predictions = to_predict.stl_predict();
        });
    })
    .expect("Computation terminated abnormally");
}

fn main() {
    let matches = App::new("Timely experiments")
        .version("0.1")
        .author("Peter Schafhalter")
        .arg(
            Arg::with_name("source")
                .long("source")
                .takes_value(true)
                .default_value("fake")
                .help("'fake' or 'redis'"),
        )
        .arg(
            Arg::with_name("global_window_size")
                .long("global_window_size")
                .takes_value(true)
                .default_value("100"),
        )
        .arg(
            Arg::with_name("global_slide_size")
                .long("global_slide_size")
                .takes_value(true)
                .default_value("100"),
        )
        .arg(
            Arg::with_name("per_key_slide_size")
                .long("per_key_slide_size")
                .takes_value(true)
                .help("JSON file containing the slide size for each key."),
        )
        .arg(
            Arg::with_name("seasonality")
                .long("seasonality")
                .takes_value(true)
                .default_value("4"),
        )
        .arg(
            Arg::with_name("prioritization")
                .long("prioritization")
                .takes_value(true)
                .default_value("fifo")
                .help("STL prioritization strategy. Either 'lifo' or 'fifo'"),
        )
        .arg(
            Arg::with_name("threads")
                .long("threads")
                .takes_value(true)
                .default_value("1")
                .help("number of threads per process."),
        )
        .arg(
            Arg::with_name("num_processes")
                .long("num_processes")
                .takes_value(true)
                .default_value("1"),
        )
        .arg(
            Arg::with_name("process_index")
                .long("process_index")
                .takes_value(true)
                .default_value("0"),
        )
        .arg(
            Arg::with_name("num_keys")
                .long("num_keys")
                .takes_value(true)
                .default_value("10")
                .help("only set this if using fake source"),
        )
        .arg(
            Arg::with_name("timesteps")
                .long("timesteps")
                .takes_value(true)
                .default_value("10000")
                .help("only set this if using fake source"),
        )
        .arg(
            Arg::with_name("send_rate")
                .long("send_rate")
                .takes_value(true)
                .default_value("1000")
                .help("in Hz. Only set this if using fake source"),
        )
        .get_matches();

    let source = matches.value_of("source").unwrap();
    let global_window_size = matches
        .value_of("global_window_size")
        .unwrap()
        .parse()
        .unwrap();
    let global_slide_size = matches
        .value_of("global_slide_size")
        .unwrap()
        .parse()
        .unwrap();
    let seasonality = matches.value_of("seasonality").unwrap().parse().unwrap();
    let prioritization: String = matches.value_of("prioritization").unwrap().parse().unwrap();
    let threads = matches.value_of("threads").unwrap().parse().unwrap();
    let num_processes: usize = matches.value_of("num_processes").unwrap().parse().unwrap();
    let process_index: usize = matches.value_of("process_index").unwrap().parse().unwrap();
    let num_keys = matches.value_of("num_keys").unwrap().parse().unwrap();
    let timesteps = matches.value_of("timesteps").unwrap().parse().unwrap();
    let send_rate = matches.value_of("send_rate").unwrap().parse().unwrap();

    let per_key_slide_size = if let Some(filename) = matches.value_of("per_key_slide_size") {
        Some(timely_experiments::parse_per_key_slide_size(filename))
    } else {
        None
    };

    run_experiment(
        source,
        global_window_size,
        global_slide_size,
        per_key_slide_size,
        seasonality,
        &prioritization,
        num_keys,
        timesteps,
        send_rate,
        threads,
        process_index,
        num_processes,
    )
}
