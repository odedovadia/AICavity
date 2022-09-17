import os
from itertools import cycle

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, recall_score, precision_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def calculate_metrics(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)

    rounded = np.round(y_pred)
    accuracy = accuracy_score(y_true, rounded)
    recall = recall_score(y_true, rounded)
    precision = precision_score(y_true, rounded)
    f1 = f1_score(y_true, rounded)
    return {'f1': f1, 'accuracy': accuracy, 'recall': recall, 'precision': precision, 'auc': auc}


def save_roc_curve(y_true, y_pred, name):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    path = os.path.join('roc_curves', name)
    if not os.path.exists(path):
        os.mkdir(path)

    np.save(os.path.join(path, 'fpr.npy'), fpr)
    np.save(os.path.join(path, 'tpr.npy'), tpr)


def plot_roc_curves():
    font = {'family': 'normal', 'weight': 'bold', 'size': 22}

    matplotlib.rc('font', **font)

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    sns.set()
    for root, dirs, files in os.walk('roc_curves'):
        if len(files) > 0:
            fpr = np.load(os.path.join(root, 'fpr.npy'))
            tpr = np.load(os.path.join(root, 'tpr.npy'))
            plt.plot(fpr, tpr, next(linecycler), label=root.split('\\')[1])
    plt.plot(fpr, fpr, label='benchmark')
    plt.title('Comparison of ROC curves')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_roc_curves()
