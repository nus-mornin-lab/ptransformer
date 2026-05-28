import torch.nn as nn
import torch
from metrics import ClsMetrics
from models.submodules import DNN
import torch.nn.init as nn_init
from simple_torch.modules import SimpleTorchModule
from simple_torch.utils import print_fn
from utils.torch_utils import param_reset
import math
import numpy as np

class FusionEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        # transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hparams.d_hidden,
                                                                                    nhead=6, batch_first=True),
                                                                                    num_layers=hparams.n_layers)

        self.classification_head = DNN(hparams.d_hidden, hparams.n_class)


        self.cls_token = nn.Parameter(torch.Tensor(hparams.d_hidden))
        nn_init.uniform_(self.cls_token, a=-1/math.sqrt(hparams.d_hidden),b=1/math.sqrt(hparams.d_hidden))

    def forward(self, x):
        """

        :param x: [batch_size, seq_len, emb_size]
        :return:
        """
        batch_size, seq_len, emb_size = x.shape

        # use [CLS] token
        cls_token_expanded = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, emb_size)
        # x_extended: [batch_size, seq_len, emb_size]
        x_extended = torch.concat([cls_token_expanded, x], dim=1)
        outputs = self.transformer_encoder(x_extended)

        row_embeddings = outputs[:, 0, :]


        outputs = self.classification_head(row_embeddings)

        return outputs.squeeze()

    
class TransMedPromptsClsSTM(SimpleTorchModule):
    def __init__(self, hparams) -> None:
        super().__init__(hparams)
        
        self.model = FusionEncoder(hparams)

        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 7.5]).cuda())
        self.metrics = ClsMetrics(hparams)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def training_step(self, x, labels):
        outputs = self(x)
        loss = self.criterion(outputs, labels.long())

        print_fn({"train_loss": loss})
        self.log({"train_loss": loss})
        return loss

    def val_step(self, x, labels):
        outputs = self(x)
        return outputs, labels

    def val_epoch_end(self, logits, y_trues):
        loss = self.criterion(logits, y_trues.long())
        acc, f1, aucroc = self.metrics(logits, y_trues.long())

        metrics = {"val_loss":loss, "val_bacc": acc, "val_f1": f1, "val_aucroc": aucroc}
        print_fn(metrics)
        self.log(metrics)


    def test_epoch_end(self, logits, y_trues):
        loss = self.criterion(logits, y_trues.long())
        acc, f1, aucroc = self.metrics(logits, y_trues.long())

        metrics = {"test_loss":loss, "test_bacc": acc, "test_f1": f1, "test_aucroc": aucroc}
        print_fn(metrics)
        self.log(metrics)


