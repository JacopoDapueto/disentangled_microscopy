
"""  Test wandb integration  """
from absl import app
from absl import flags

import os
import yaml
import wandb

from code.training.train_adagvae import train_model
from code.postprocessing.postprocess import postprocess_model
from code.visualization.visualize_model import visualize_model
from code.downstream_task.downstream_task import GBT_regressor, MLP_regressor

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment", "VAE_DRL_plankton", "Name of the experiment to run")
flags.DEFINE_string("config", "train_drl_maskedplankton", "Name of the config to run")
flags.DEFINE_string("output_directory", None, "Output directory of experiments ('{model_num}' will be"
                                                " replaced with the model index  and '{experiment}' will be"
                                                " replaced with the study name if present).")


def get_model_num(config):
    return "beta-" + str(config["beta"]) + "+" +  "random_seed-" + str(config["random_seed"])


def read_config():
    # Read YAML file
    with open(f"configs/{FLAGS.config}/config.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded


def sweep(unused_args):

    # Initialize sweep by passing in configs.
    # (Optional) Provide a name of the project.
    sweep_id = wandb.sweep(sweep=read_config(), project="adagvae_training")

    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=1)


def main():

    run = wandb.init(project="adagvae_training",
                     notes="Train VAE on Texture-dSprites",
                     tags=["VAE", "Texture-dSprites", "DRL"]
                     )

    assert wandb.run is not None


    # Set correct output directory. and check they already exist
    if FLAGS.output_directory is None:
        output_directory = os.path.join("output", "{experiment}", "{model_num}")
    else:
        output_directory = FLAGS.output_directory

    model_num = get_model_num(run.config)
    run.name = model_num

    # Insert model number and study name into path if necessary.
    output_directory = output_directory.format(model_num=str(run.name),
                                               experiment=str(FLAGS.experiment))


    # make experiment directory
    if not os.path.exists(output_directory):
        # if the demo_folder directory is not present then create it.
        os.makedirs(output_directory)
    else:
        raise FileExistsError("Experiment folder exists")


    # train model
    print(f"SWEEP: {run.id}")
    print(run.config)

    with open(os.path.join(output_directory, 'config.yaml'), 'w') as f:
        _ = yaml.dump(run.config, f)

    train_model(output_directory, run.config)
    postprocess_model(output_directory, run.config)
    run.config["path_to_representation"] = os.path.join(output_directory, "postprocess")

    # visualization
    visualize_model(output_directory, run.config)

    # run plankton classification
    gbt_classifier = GBT_regressor(mode=None, path= run.config["path_to_representation"])
    mlp_classifier = MLP_regressor(mode=None, path= run.config["path_to_representation"])

    print("===============START DOWNSTREAMTASK===============")
    scores = gbt_classifier.get_score()
    scores = mlp_classifier.get_score()

    print("===============END DOWNSTREAMTASK===============")
    run.finish()


if __name__ == "__main__":
    app.run(sweep)
