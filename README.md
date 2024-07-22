# Disentangled representations of plankton images for oceanic ecosystems monitoring
Official repository of "Disentangled representations of plankton images for oceanic ecosystems monitoring"


## Set up
### Install
The code was developed and tested with Python 3.10 and the main dependencies are Pytorch 2.0.0 and Cuda 11.7. We use Wandb to log scores and models.

Set up the environment and then install the dependencies with
```
pip install -r requirements.txt
```

### Download and prepare datasets

1. Set the environment variable `DISENTANGLEMENT_LIB_DATA` to this path, for example by adding

```
export DISENTANGLEMENT_LIB_DATA=<path to the data directory>
```
2. Download all the necessary datasets that can be found in the following links
   [Lenseless](https://ibm.ent.box.com/v/PlanktonData)
   [WHOI40](https://ibm.ent.box.com/v/PlanktonData)
   [WHOI15](https://github.com/Malga-Vision/Anomaly-detection-in-feature-space-for-detecting-changes-in-phytoplankton-populations)

4. Unzip the compressed files and put in DISENTANGLEMENT_LIB_DATA

5. Postprocess the dataset with the commands
```
python code/dataset/preprocessing_lensless.py
python code/dataset/preprocessing_whoi15.py
python code/dataset/preprocessing_whoi40.py
```

## How to reproduce 

`./config` folder contains the .yaml files to reproduce the experiments


### Extract deep features from the backbone
To extract the features $\Phi$ executes:
```
python dlib_extract_from_backbone.py --dataset <name of the dataset>
```
The representation is saved in DISENTANGLEMENT_LIB_DATA as .npz file.

The name of the datasets to be used can be found in `./code/choose_dataset.py`


### Compute hand-crafted features
To extract the hand-crafted features of Lensless:
```
python dlib_extract_simple_features.py
```
The features are saved in DISENTANGLEMENT_LIB_DATA as `SIMPLE_FEATURES/representation.npz` file.


### Train Source models
The scripts starting with *train_* execute the training of the Source models.

```
bash ./bash_scripts/train_*.sh
```

The results will be saved in `./output` directory, organized by _experiment name_ and numbered by the _random seed_.

**Once** one experiment folder is completed aggregate the results of all random seeds with the scripts
```
python dlib_aggregate_results_experiment.py --experiment experiment_name 
```


### Transfer on Target dataset
**Once** you have trained the source models, run the scripts to execute the transfer experiments.
To transfer (without and with fine-tuning) and to evaluate the representation, run the following script:
```
python dlib_transfer_dsprites_to_plankton.py --config <name of the config folder> --experiment <name of the output directory>
```
The results will be saved in `./output` directory, organized by experiment name:
*experiment/sweeps* and each sweep is further divided into `before` and `after` folders (meaning w/o or w/ finetuning ).

**Once** one experiment folder is completed aggregate the results of all random seeds with the scripts
```
python dlib_group_results_scores.py --experiment experiment_name 
```
For each score of interest (e.g. accuracy) a .json file is created inside the experiment folder reporting mean and std.

### Open set experiment
To evaluate the model on the open-set setting, run the following script with default parameters:

```
python dlib_evaluate_open_set.py
```


### Without disentanglement
To evaluate the classifiers directly on $\Phi$, run the script with default parameters:

```
python dlib_compute_baseline.py
```


### Visualizations on Target dataset

To reproduce the visualizations run the following scripts
```
python dlib_visualize_latent_space.py --config <name of the config folder> --experiment <name of the output directory> --model_num <name of the sweep to visualize>
python dlib_visualize_latent_space_with_handcrafted.py --config <name of the config folder> --experiment <name of the output directory> --model_num <name of the sweep to visualize>
python dlib_visualize_latent_space_open_set.py --config <name of the config folder> --experiment <name of the output directory> --model_num <name of the sweep to visualize>
```




