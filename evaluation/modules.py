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
        self.loss_modules = nn.ModuleDict(
            {
                "KL_Divergence": torchmetrics.KLDivergence(),
                # "CC": torchmetrics.PearsonCorrCoef(num_outputs=batch_size),
                "L1": nn.L1Loss(),
            }
        )

    def compute_loss(self, loss_function_name, pred, gt):
        loss = None
        assert pred.size() == gt.size()
        assert len(pred.size()) != 4
        if len(pred.size()) == 4:  # BxCXHxW
            assert pred.size(0) == self.batch_size
            pred = pred.permute((1, 0, 2, 3))
            gt = gt.permute((1, 0, 2, 3))

            for i in range(pred.size(0)):
                if loss == None:
                    loss = self.loss_modules[loss_function_name](
                        pred[i].reshape((pred.size(1), -1)),
                        gt[i].reshape((gt.size(1), -1)),
                    )
                else:
                    loss += self.loss_modules[loss_function_name](
                        pred[i].reshape((pred.size(1), -1)),
                        gt[i].reshape((gt.size(1), -1)),
                    )

            loss /= pred.size(0)
            return loss
        print("Loss Arguments", pred.reshape((pred.size(0), -1)).shape, gt.reshape((gt.size(0), -1)).shape)
        if loss_function_name == 'similarity':
            loss = self.similarity(pred, gt)
        elif loss_function_name == "CC":
            loss = self.pearson_coeff(pred, gt)
        else:
            loss = self.loss_modules[loss_function_name](
                pred.reshape((pred.size(0), -1)), gt.reshape((gt.size(0), -1))
            )
        print(f"Loss {loss_function_name}: value {loss}")
        return loss
    
    def _normalize_matrix(self, mat):
        # normalize the salience map (as done in MIT code)
        batch_size = mat.size(0)
        w = mat.size(1)
        h = mat.size(2)
        
        min_mat = torch.min(mat.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)
        max_mat = torch.max(mat.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)

        norm_mat = (mat - min_mat)/(max_mat-min_mat*1.0)
        return norm_mat
    
    def pearson_coeff(self, pred, gt):
        assert pred.size() == gt.size()
        batch_size = pred.size(0)
        w = pred.size(1)
        h = pred.size(2)

        mean_pred = torch.mean(pred.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
        std_pred = torch.std(pred.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

        mean_gt = torch.mean(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
        std_gt = torch.std(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

        pred = (pred - mean_pred) / std_pred
        gt = (gt - mean_gt) / std_gt

        ab = torch.sum((pred * gt).view(batch_size, -1), 1)
        aa = torch.sum((pred * pred).view(batch_size, -1), 1)
        bb = torch.sum((gt * gt).view(batch_size, -1), 1)

        return torch.mean(ab / (torch.sqrt(aa*bb)))
    
    def similarity(self, pred, gt):
        batch_size = pred.size(0)
        w = pred.size(1)
        h = pred.size(2)

        pred = self._normalize_matrix(pred)
        gt = self._normalize_matrix(gt)
        
        sum_pred = torch.sum(pred.view(batch_size, -1), 1)
        expand_pred = sum_pred.view(batch_size, 1, 1).expand(batch_size, w, h)
        
        assert expand_pred.size() == pred.size()

        sum_gt = torch.sum(gt.view(batch_size, -1), 1)
        expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)

        pred = pred/(expand_pred*1.0)
        gt = gt / (expand_gt*1.0)

        pred = pred.view(batch_size, -1)
        gt = gt.view(batch_size, -1)
        return torch.mean(torch.sum(torch.min(pred, gt), 1))