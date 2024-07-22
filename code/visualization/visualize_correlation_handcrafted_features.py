
import os
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from code.dataset.plakton_padded import PLANKTON_ORIGINAL_PATH, PLANKTON_PADDED224_PATH

def save_distplot(x, name, path):
    data = {name: [elem for elem in x]}

    with sns.axes_style("whitegrid"):
        sns.displot(data=data, x=name) # , kde=True


    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.clf()



def quantize_data(data, num_bins=10):
    """
    Quantizes the data into the specified number of bins.

    Parameters:
    - data: The input data to be quantized (a list or numpy array).
    - num_bins: The number of bins to quantize the data into.

    Returns:
    - quantized_data: The quantized data.
    """
    quantized_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        min_val = np.min(data[:,i])
        max_val = np.max(data[:,i])

        # Compute the step size for the quantization
        step_size = (max_val - min_val) / (num_bins - 1)

        # Quantize the data
        quantized_data[:,i] = np.round((data[:,i] - min_val) / step_size) * step_size + min_val



    return quantized_data




def compute_features_corrcoeff(rep, handcrafted_rep):

    #quantize
    rep = quantize_data(rep, 20)
    handcrafted_rep =  quantize_data(handcrafted_rep, 20)

    n_features = rep.shape[1]


    df_rep = pd.DataFrame(rep)
    df_handcrafted_rep = pd.DataFrame(handcrafted_rep)
    df = pd.concat([df_rep, df_handcrafted_rep], axis=1)

    corr = df.corr(method='spearman').values # spearman # kendall #pearson

    corr = np.abs(corr)
    corr = corr[n_features:, :n_features]
    return corr


def save_plotly( matrix , title, path, log_key="", use_wandb=False):


    fig = px.imshow(matrix, text_auto=False,  aspect="auto", color_continuous_scale='Blues', origin='lower', labels=dict(x="Deep features", y="Hand-crafted features", color="Correlation"), zmax=1., zmin=0.)

    fig.update_layout(title=title)

    if use_wandb:
        wandb.log({log_key: wandb.Plotly(fig)})
    else:
        fig.write_html(path)



def create_visualization_directory(directory):

    process_dir = os.path.join(directory, "visualization")

    # make experiment directory
    if not os.path.exists(process_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(process_dir)


    return process_dir


def visualize_correlation(directory, args):

    directory = create_visualization_directory(directory)

    # load representation
    path_to_representation = args["path_to_representation"]
    rep = np.load(os.path.join(path_to_representation, "representations.npz"))

    # load handcrafted representation
    handcrafted_rep = np.load(os.path.join(PLANKTON_ORIGINAL_PATH, "representations.npz"))


    x_train, x_val, x_test = rep["train"], rep["val"], rep["test"]
    handcrafted_x_train, handcrafted_x_val, handcrafted_x_test = handcrafted_rep["train"], handcrafted_rep["val"], handcrafted_rep["test"]

    for name, i in zip(["Avg color", "Dominant color", "Area", "Orientation"], range(handcrafted_x_test.shape[1])):
        save_distplot(handcrafted_x_test[:, i], name, path=os.path.join(directory, f"{name}_dist.png"))


    corr = compute_features_corrcoeff(x_train, handcrafted_x_train)
    save_plotly(corr, title="Correlation",  path=os.path.join(directory, "train_features_corr.html"))

    corr = compute_features_corrcoeff(x_val, handcrafted_x_val)
    save_plotly(corr, title="Correlation", path=os.path.join(directory, "val_features_corr.html"))

    corr = compute_features_corrcoeff(x_test, handcrafted_x_test)
    save_plotly(corr, title="Correlation", path=os.path.join(directory, "test_features_corr.html"))

    # load simple handcrafted representation
    handcrafted_rep = np.load(os.path.join(PLANKTON_PADDED224_PATH, "SIMPLE_FEATURES", "representations.npz"))

    x_train, x_val, x_test = rep["train"], rep["val"], rep["test"]
    handcrafted_x_train, handcrafted_x_val, handcrafted_x_test = handcrafted_rep["train"], handcrafted_rep["val"], \
    handcrafted_rep["test"]

    corr = compute_features_corrcoeff(x_train, handcrafted_x_train)
    save_plotly(corr, title="Correlation", path=os.path.join(directory, "train_simple_features_corr.html"))

    corr = compute_features_corrcoeff(x_val, handcrafted_x_val)
    save_plotly(corr, title="Correlation", path=os.path.join(directory, "val_simple_features_corr.html"))

    corr = compute_features_corrcoeff(x_test, handcrafted_x_test)
    save_plotly(corr, title="Correlation", path=os.path.join(directory, "test_simple_features_corr.html"))

