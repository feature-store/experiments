import wandb
import configparser
import os


def log_questions(run, config):
    # log questions file
    artifact = wandb.Artifact("questions", type='dataset')
    artifact.add_file(config["files"]["raw_questions_file"])
    artifact.add_file(config["files"]["questions_file"])
    run.log_artifact(artifact)

def log_files(run, config):
    # log files
    artifact = wandb.Artifact("files", type='dataset')
    artifact.add_file(config["files"]["changes_file"])
    artifact.add_file(config["files"]["titles_file"])
    artifact.add_file(config["files"]["edits_file"])
    run.log_artifact(artifact)

def log_pageview(run, config):
    # log pageview
    artifact = wandb.Artifact("pageviews", type='dataset')
    artifact.add_file(config["files"]["raw_pageview_file"])
    artifact.add_file(config["files"]["pageview_file"])
    artifact.add_file(config["files"]["timestamp_weights_file"])
    run.log_artifact(artifact)

def log_simulation(run, config):
    # log simulation data 
    artifact = wandb.Artifact("simulation", type='dataset')
    artifact.add_file(config["simulation"]["stream_edits_file"])
    artifact.add_file(config["simulation"]["stream_questions_file"])
    artifact.add_file(config["simulation"]["init_data_file"])
    run.log_artifact(artifact)

def log_plans(run, config, plan_dir):
    artifact = wandb.Artifact("plans", type='dataset')
    artifact.add_file(config["simulation"]["optimal_plan_file"])
    artifact.add_dir(plan_dir)
    run.log_artifact(artifact)

def log_plan_data(run, config, plan_name, plan_path):
    artifact = wandb.Artifact(plan_name, type='dataset')
    artifact.add_folder(plan_path)
    run.log_artifact


def log_experiment(run, config):
    # log experiment output
    artifact = wandb.Artifact("prediction_results", type='dataset')
    files = os.listdir(config["directory"]["dpr_dir"])
    for filename in files: 
        if "plan-" in filename and '.json' in filename:
            artifact.add_file(os.path.join(config["directory"]["dpr_dir"], filename))
    run.log_artifact(artifact)

if __name__ == "__main__":

    print("Running wandb logging on data")
    run = wandb.init(job_type="dataset-creation", project="wiki-workload")

    # configuration file
    config = configparser.ConfigParser()
    config.read("config.yml")
    
    log_questions(run, config)
    log_files(run, config)
    log_pageview(run, config)
    log_simulation(run, config)
    log_experiment(run, config)

