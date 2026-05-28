from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics 
from sklearn.metrics import balanced_accuracy_score


class ClsMetrics(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.n_class = hparams.n_class
        if hparams.n_class == 2:
            # self.acc_metric = torchmetrics.Accuracy(task="binary")
            self.f1_metric = torchmetrics.F1Score(task="binary")
            self.aucroc_metric = torchmetrics.AUROC(task="binary")
        else:
            # self.acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=hparams.n_class)
            self.f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=hparams.n_class)
            self.aucroc_metric = torchmetrics.AUROC(task="multiclass", num_classes=hparams.n_class)            

        self.bacc_metric = balanced_accuracy_score

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, 1)

        # acc = self.acc_metric(preds, targets)
        bacc = self.bacc_metric(targets.cpu().numpy(), preds.cpu().numpy())
        f1 = self.f1_metric(preds, targets)

        if self.n_class == 2:
            aucroc = self.aucroc_metric(probs[:, 1], targets)
        else:
            aucroc = self.aucroc_metric(probs, targets)

        return bacc, f1, aucroc
