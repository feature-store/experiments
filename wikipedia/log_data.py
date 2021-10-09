import wandb
import configparser
import os




if __name__ == "__main__":

    print("Running wandb logging on data")
    run = wandb.init(job_type="dataset-creation", project="wiki-workload")

    # configuration file
    config = configparser.ConfigParser()
    config.read("config.yml")

    # log files
    artifact = wandb.Artifact("files", type='dataset')
    artifact.add_file(config["files"]["changes_file"])
    artifact.add_file(config["files"]["titles_file"])
    artifact.add_file(config["files"]["edits_file"])
    run.log_artifact(artifact)

    # log pageview
    artifact = wandb.Artifact("pageviews", type='dataset')
    artifact.add_file(config["files"]["raw_pageview_file"])
    artifact.add_file(config["files"]["pageview_file"])
    artifact.add_file(config["files"]["timestamp_weights_file"])
    run.log_artifact(artifact)

    # log questions file
    artifact = wandb.Artifact("questions", type='dataset')
    artifact.add_file(config["files"]["raw_questions_file"])
    artifact.add_file(config["files"]["questions_file"])
    run.log_artifact(artifact)

    # log simulation data 
    artifact = wandb.Artifact("simulation", type='dataset')
    artifact.add_file(config["simulation"]["stream_edits_file"])
    artifact.add_file(config["simulation"]["stream_questions_file"])
    artifact.add_file(config["simulation"]["init_data_file"])
    run.log_artifact(artifact)


    # log experiment output
    artifact = wandb.Artifact("prediction_results", type='dataset')
    files = os.listdir(config["directory"]["dpr_dir"])
    for filename in files: 
        if "plan-" in filename and '.json' in filename:
            artifact.add_file(os.path.join(config["directory"]["dpr_dir"], filename))
    run.log_artifact(artifact)
