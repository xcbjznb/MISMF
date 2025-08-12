from math import inf

import torch
from sklearn.utils.multiclass import type_of_target
from torch import nn
from torch.autograd import Function

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_score, recall_score, roc_auc_score, log_loss, \
    confusion_matrix

np.set_printoptions(threshold=inf)

import torch
import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


def get_precision(output, target, average):
    if torch.is_tensor(output):
        output = (output.view(-1)).data.cpu().numpy()

    if torch.is_tensor(target):
        target = (target.view(-1)).data.cpu().numpy()

    return precision_score(target, output, average=average)


def get_accuracy(output, target):
    if torch.is_tensor(output):
        output = (output.view(-1)).data.cpu().numpy()

    if torch.is_tensor(target):
        target = (target.view(-1)).data.cpu().numpy()

    return accuracy_score(target, output)


def get_f1_score(output, target, average):
    if torch.is_tensor(output):
        output = (output.view(-1)).data.cpu().numpy()

    if torch.is_tensor(target):
        target = (target.view(-1)).data.cpu().numpy()

    return f1_score(target, output, average=average)


def get_roc_curve(output, target):
    if torch.is_tensor(output):
        output = (output.view(-1)).data.cpu().numpy()

    if torch.is_tensor(target):
        target = (target.view(-1)).data.cpu().numpy()

    return roc_curve(target, output)


def get_roc_auc_score(output, target):
    if torch.is_tensor(output):
        output = (output.view(-1)).data.cpu().numpy()

    if torch.is_tensor(target):
        target = (target.view(-1)).data.cpu().numpy()

    return roc_auc_score(target, output)


def get_confusion(output, target):
    if torch.is_tensor(output):
        output = (output.view(-1)).data.cpu().numpy()

    if torch.is_tensor(target):
        target = (target.view(-1)).data.cpu().numpy()

    return confusion_matrix(target, output).ravel()


def get_sensitivity(output, target, average):
    if torch.is_tensor(output):
        output = (output.view(-1)).data.cpu().numpy()

    if torch.is_tensor(target):
        target = (target.view(-1)).data.cpu().numpy()

    return recall_score(target, output, average=average)


def get_MIoU(output, target):
    hist = get_confusion(output, target)
    hist = hist.reshape(2, 2)
    a = np.diag(hist)
    b = hist.sum(1)
    c = hist.sum(0)
    d = np.diag(hist)

    return a / (b + c - d)




