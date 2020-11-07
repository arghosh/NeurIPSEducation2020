import json
import numpy as np
from sklearn import metrics
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data


def dump_json(path_, data):
    with open(path_, 'w') as fh:
        json.dump(data, fh, indent=2)
    return data


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)
