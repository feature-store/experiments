import wandb
import configparser
import os


def log_experiment(run, config):
    # log experiment output
    artifact = wandb.Artifact("results", type='dataset')
    artifact.add_dir("/data/wooders/stl/results")
    run.log_artifact(artifact)

def log_train(run, config):
    # log experiment output
    artifact = wandb.Artifact("yahoo_train_data", type='dataset')
    artifact.add_dir("yahoo_train_data")
    run.log_artifact(artifact)

def log_eval(run, config):
    # log experiment output
    artifact = wandb.Artifact("yahoo_eval_data", type='dataset')
    artifact.add_dir("yahoo_eval_data")
    run.log_artifact(artifact)

def log_oracle(run, config):
    # log experiment output
    artifact = wandb.Artifact("oracle", type='dataset')
    artifact.add_dir("oracle")
    run.log_artifact(artifact)



if __name__ == "__main__":

    print("Running wandb logging on data")
    run = wandb.init(job_type="dataset-creation", project="stl")

    # configuration file
    config = configparser.ConfigParser()
    config.read("config.yml")
    
    log_experiment(run, config)
    log_train(run, config)
    log_eval(run, config)
    log_oracle(run, config)

