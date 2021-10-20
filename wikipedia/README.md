# Wikipedia Experiment Pipeline

### Configuration
Update `config.yml` 

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


## Offline Simulation Pipeline
Download the data with `./download_data.sh` (warning: 100s of GBs) and update `config.yml`.

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


