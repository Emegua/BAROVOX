import torch
import core.dataloader
import numpy as np
import torchaudio
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
import torch.nn as nn
import seaborn as sn
import pandas as pd

import matplotlib.pyplot as plt
import wandb



def number_of_correct(pred, target):
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    return tensor.argmax(dim=-1)

def log_confusion_matrix(y_pred, y_true, epoch=0, labels=None):
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in range(len(labels))],
                    columns = [i for i in range(len(labels))])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    wandb.log({f"confusion_matrix_{epoch}": wandb.Image(plt)})
    plt.clf()







