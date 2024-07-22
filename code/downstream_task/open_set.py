"""Downstream classification task."""

import os
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import json

import wandb

from code.downstream_task.metric import Metric


def predict(model, threshold, x):

    distances, _ = model.kneighbors(x)

    avg_distances = distances.mean(axis=1)
    preds = (avg_distances > threshold).astype(np.int32)

    return preds



def make_confusion_matrix(y_true, y_pred, labels, title, path, wandb=True):
    matrix = confusion_matrix(y_true, y_pred)
    matrix = matrix.astype(np.float64)

    n_labels = len(labels)


    # normalize confusion matrix
    for i in range(int(n_labels)):
        # M[i,j] stands for Element of real class i was classified as j
        sum = np.sum(matrix[i, :])
        matrix[i, :] = matrix[i, :] / sum

    known_acc, unknown_acc = matrix[0,0], matrix[1, 1]


    df_cm = pd.DataFrame(matrix, labels, labels)

    # matrix of bool values, True if different from Zero
    annot_pd = df_cm.applymap(lambda x: "{:.2%}".format(
        x) if round(x, 3) != 0.000 else '')

    mean_acc = np.array([matrix[i, i]
                         for i in range(n_labels)]).sum() / n_labels

    std_acc = np.std(np.array([matrix[i, i]
                               for i in range(n_labels)]))

    fig = plt.figure(figsize=(10, 7))
    fig.tight_layout()

    sn.set(font_scale=0.55)  # label size
    sn.heatmap(df_cm, annot=annot_pd, annot_kws={
        "size": 9}, fmt='s', vmin=0, vmax=1, cmap="Blues", cbar=False)  # font size

    plt.title(
        "Mean acc: {:.2%} - Std: {:.2} - {}".format(mean_acc, std_acc, title))

    plt.subplots_adjust(bottom=0.28)




    if not wandb:
        plt.savefig(path, dpi=300, bbox_inches='tight')

        print("Save plot in " + path)
        # clear the matplotlib figure to free memory
        #plt.close(fig)
        return fig, mean_acc, known_acc, unknown_acc

    return fig, mean_acc, known_acc, unknown_acc


def make_predictor(name):

    if name == "knn":
        return NearestNeighbors(n_neighbors=5)
    else:
        raise "Name of predictor not exists!"


def _compute_model(x_train, x_test, predictor):
    """Compute average accuracy for train and test set."""
    train_acc = 0.0
    test_acc = 0.0

    # concat train and val
    x_train = np.concatenate((x_train, x_test), axis=0)

    model = predictor
    model.fit(x_train)


    return  model


def compute_downstream_task(representation, classes, predictor):

    scores = {}

    x_train, x_val, x_test = representation["train"], representation["val"], representation["test"]

    predictor_model = make_predictor(predictor)

    model = _compute_model(x_train, x_val, predictor_model)

    train_distances, _ = model.kneighbors(x_train)
    val_distances, _ = model.kneighbors(x_val)
    test_distances,_ = model.kneighbors(x_test)
    scores["train"], scores["val"], scores["test"]  = train_distances.mean(axis=1).mean(), val_distances.mean(axis=1).mean(), test_distances.mean(axis=1).mean()
    return scores, model




class KNN_open_set(Metric):

    def __init__(self,  mode, **kwargs):

        super(KNN_open_set, self).__init__(**kwargs)
        self.mode = mode
        self.predictor = "knn"
        self.name = "open_set_knn"


    def get_score(self, class_to_evaluate, class_name ):
        ''' Return the score '''

        output_directory = os.path.join(self.path, self.predictor)

        # make experiment directory
        if not os.path.exists(output_directory):
            # if the demo_folder directory is not present then create it.
            os.makedirs(output_directory)


        # load representation
        rep = np.load(os.path.join(self.path, "representations.npz") )


        csv = np.load(os.path.join(self.path, "classes.npz") )

        test_labels_name = []
        with open(os.path.join(self.path, 'TEST_labels_name.txt')) as file:
            for line in file:
                test_labels_name.append(line)  # storing everything in memory!

        train_labels_name = []
        with open(os.path.join(self.path, 'TRAIN_labels_name.txt')) as file:
            for line in file:
                train_labels_name.append(line)  # storing everything in memory!

        scores, model = compute_downstream_task(rep, csv, predictor=self.predictor)


        with open(os.path.join(output_directory, 'distances.json'), 'w') as fp:
            json.dump(scores, fp)

        x_train, x_val, x_test = rep["train"], rep["val"], rep["test"]
        y_train, y_val, y_test = csv["train"].astype(np.int32), csv["val"].astype(np.int32), csv["test"].astype(np.int32)

        # modify the labels into ["KNOWN", "UNKOWN"]
        labels_name = ["Others", class_name]
        # train and validation are only knowns
        y_train, y_val, y_test = np.where(y_train != class_to_evaluate, 0, 0), np.where(y_val != class_to_evaluate, 0, 0), np.where(y_test != class_to_evaluate, 0, 1)


        # compute threshold
        distances, _ = model.kneighbors(x_train)

        avg_distances = distances.mean(axis=1)

        threshold =  np.percentile(avg_distances, 95)


        fig_train, train_balanced_acc, train_known_acc, train_unknown_acc = make_confusion_matrix(y_train, predict(model, threshold, x_train), labels_name, "Training set", os.path.join(output_directory, self.predictor + "_train_confusion.png"), self.wandb)
        fig_val, val_balanced_acc, val_known_acc, val_unknown_acc = make_confusion_matrix(y_val, predict(model, threshold, x_val), labels_name, "Validation set", os.path.join(output_directory, self.predictor +"_val_confusion.png"), self.wandb)
        fig_test, test_balanced_acc, test_known_acc, test_unknown_acc = make_confusion_matrix(y_test, predict(model, threshold, x_test), labels_name, "Test set", os.path.join(output_directory, self.predictor +"_test_confusion.png"), self.wandb)



        with open(os.path.join(output_directory, 'balanced_average_binary_accuracy.json'), 'w') as fp:
            json.dump({"train": train_balanced_acc, "val": val_balanced_acc, "test": test_balanced_acc}, fp)

        with open(os.path.join(output_directory, 'balanced_single_class_binary_accuracy.json'), 'w') as fp:
            json.dump({"train_known": train_known_acc,
                       "val_known": val_known_acc,
                       "test_known": test_known_acc,
                       "train_unknown": train_unknown_acc,
                       "val_unknown": val_unknown_acc,
                       "test_unknown": test_unknown_acc
                       }, fp)


        # save model
        with open(os.path.join(output_directory, self.predictor + 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)

        # clear the matplotlib figure to free memory
        plt.close(fig_train)
        plt.close(fig_val)
        plt.close(fig_test)




        return scores









