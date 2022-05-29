# Wikipedia Experiment Pipeline

*MUST USE PYTHON 3.6* (due to DPR dependency) 

The Wikipedia pipeline uses data collected from the Wikipedia RecentChanges and PageView API to simulate maintaining document embeddings as documents are edited and queried by a downstream retrival task. 

1. *Offline Simulation Pipeline* - The offline simulation pipeline in the `offline/` folder usegenerate plans under different simulation settings and evaluates overall prediction quality. 
2. *Online Pipeline* - The online pipeline streams simulation data into ralf, which updates an in-memory table of embeddings and responds to client queries. 

## Offline Simulation Pipeline

### Download data
Download the data with `./download_data.sh` (warning: 100s of GBs) and update `config.yml` to match the paths.

### Generating simulation data
Run parts of the pipeline using flags: 
```
python generate_data.py \
	--run_query_recentchanges # query wikipedia recentchanges api
	--run_query_doc_versions # query wikipedia docs api
	--run_recent_changes # process raw changes data into changes.csv file
	--run_parse_docs # process raw doc data with wikiparser
	--run_get_questions # process raw questions into questions.csv
	--run_get_pageviews # process raw pageview data into pageviews.csv
	--run_generate_diffs # compute diffs between different version
	--run_generate_simulation_data # generate simulation data
	--run_check_dataset # check dataset
	--run_generate_embeddings # embed documents 
```
To update simulation data, make sure you have the embeddings and diffs already download, and run: 
```
python generate_data.py --run_generate_simulation_data --run_get_questions --run_check_dataset
```

### Running simulations 
Run the simulation in stages to go from raw Wikipedia API data to simulation results: 

```
./run_0_generate_data.sh # generate simulation data from questions.csv file
./run_1_generate_plan.sh # run simulations to generate plan
./run_2_prepare_data.sh # use plan to determine questions / embedding versions at each timestep 
./run_3_run_predictions.sh # run DPR model on embeddings 
./run_4_run_optimal_predictons.sh # generate optimal predictions 
```

### Logging Data 
To save the current data, run 
```
python log_data.py
```

### Logging Experiments
TODO
 
## Online Pipeline (ralf)
(NOTE: incomplete) 
Run the server 
```
python wiki_server.py
```
Run the client 
```
python wiki_client.py 
```


