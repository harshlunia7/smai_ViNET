import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import kl_divergence
from torchmetrics import Metric


class Evaluation_Metric(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.loss_functions = nn.ModuleDict(
            {
                "KL_Divergence": torchmetrics.KLDivergence(),
                "CC": torchmetrics.PearsonCorrCoef(num_outputs=2),
                "L1": nn.L1Loss(),
                "AUROC": torchmetrics.AUROC(),
            }
        )

    def compute_loss(self, loss_function_name, pred, gt):
        loss = None
        assert pred.size() == gt.size()
        if len(pred.size()) == 4:  # BxCXHxW
            assert pred.size(0) == self.batch_size
            pred = pred.permute((1, 0, 2, 3))
            gt = gt.permute((1, 0, 2, 3))

            for i in range(pred.size(0)):
                if loss == None:
                    loss = self.loss_functions[loss_function_name](
                        pred[i].reshape((pred.size(1), -1)),
                        gt[i].reshape((gt.size(1), -1)),
                    )
                else:
                    loss += self.loss_functions[loss_function_name](
                        pred[i].reshape((pred.size(1), -1)),
                        gt[i].reshape((gt.size(1), -1)),
                    )

            loss /= pred.size(0)
            return loss
        return self.loss_functions[loss_function_name](
            pred.reshape((pred.size(0), -1)), gt.reshape((gt.size(0), -1))
        )
