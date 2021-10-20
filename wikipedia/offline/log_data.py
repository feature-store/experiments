import wandb
import configparser
import os


def log_questions(run, config):
    # log questions file
    artifact = wandb.Artifact("questions", type='dataset')
    artifact.add_file(config["files"]["raw_questions_file"])
    artifact.add_file(config["files"]["questions_file"])
    artifact.add_file(config["files"]["test_questions_file"])
    artifact.add_file(config["files"]["train_questions_file"])
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
    artifact.add_dir(config["simulation"]["weights_dir"])
    run.log_artifact(artifact)

def log_plans(run, config, plan_dir):
    artifact = wandb.Artifact("plans", type='dataset')
    artifact.add_file(config["simulation"]["optimal_plan_file"])
    artifact.add_dir(plan_dir)
    run.log_artifact(artifact)


def log_experiment(run, config):
    # log experiment output
    artifact = wandb.Artifact("prediction_results", type='dataset')
    artifact.add_dir("/data/wooders/wikipedia/predictions")
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

