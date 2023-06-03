import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes, confusion_mat=False):
    targets = targets.numpy()
    _, pred = torch.max(logits, dim=1)
    pred = pred.numpy()
    acc = accuracy_score(targets, pred)
    f1 = f1_score(targets, pred, average='weighted')

    probs = F.softmax(logits, dim=1)
    probs = probs.numpy()
    if len(np.unique(targets)) != num_classes:
        roc_auc = 0
    else:
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true=targets, y_score=probs[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
        else:
            binary_labels = label_binarize(targets, classes=[i for i in range(num_classes)])
            valid_classes = np.where(np.any(binary_labels, axis=0))[0]
            binary_labels = binary_labels[:, valid_classes]
            valid_cls_probs = probs[:, valid_classes]
            fpr, tpr, _ = roc_curve(y_true=binary_labels.ravel(), y_score=valid_cls_probs.ravel())
            roc_auc = auc(fpr, tpr)
    if confusion_mat:
        mat = confusion_matrix(targets, pred)
        return acc, f1, roc_auc, mat
    return acc, f1, roc_auc


def plot_confusion_matrix(cmtx, num_classes, class_names=None, title='Confusion matrix', normalize=False,
                          cmap=plt.cm.Blues):
    if normalize:
        cmtx = cmtx.astype('float') / cmtx.sum(axis=1)[:, np.newaxis]
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure()
    plt.imshow(cmtx, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    fmt = '.2f' if normalize else 'd'
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        plt.text(j, i, format(cmtx[i, j], fmt), horizontalalignment="center",
                 color="white" if cmtx[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure


def draw_metrics(ts_writer, name, num_class, loss, acc, auc, mat, f1, step):
    ts_writer.add_scalar("{}/loss".format(name), loss, step)
    ts_writer.add_scalar("{}/acc".format(name), acc, step)
    ts_writer.add_scalar("{}/auc".format(name), auc, step)
    ts_writer.add_scalar("{}/f1".format(name), f1, step)
    if mat is not None:
        ts_writer.add_figure("{}/confusion mat".format(name),
                             plot_confusion_matrix(cmtx=mat, num_classes=num_class), step)
