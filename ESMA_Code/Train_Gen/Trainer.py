import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .gaussian_smoothing import get_gaussian_kernel

class GeneratorTrainer(nn.Module):
    def __init__(self, model,source_model,target_match,num_target,eps,normalize=True,mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]):
        super().__init__()
        self.eps = eps
        self.model = model
        self.source_model = source_model
        self.num_labels = num_target
        self.match_dict = target_match
        self.mean = mean
        self.std = std
        self.normalize = normalize
    def norm(self, t):
        if self.normalize:
            t[:, 0, :, :] = (t[:, 0, :, :] - self.mean[0])/self.std[0]
            t[:, 1, :, :] = (t[:, 1, :, :] - self.mean[1])/self.std[1]
            t[:, 2, :, :] = (t[:, 2, :, :] - self.mean[2])/self.std[2]
        else:
            pass
        return t
    def forward(self, x, src_labels):

        kernel = get_gaussian_kernel(kernel_size=3, pad=2, sigma=1).to(x.device)
        self.source_model.eval()
        target_label = torch.randint(self.num_labels, size=(x.shape[0], ), device=x.device)
        x_match = []
        for i in range(target_label.shape[0]):
            x_match.append(self.match_dict[target_label[i].item()])

        x_match = torch.tensor(np.array([item.cpu().detach().numpy() for item in x_match])).to(x.device)
        mask = torch.ne(src_labels,target_label).long()
        
        perturbated_imgs = kernel(self.model(x,target=target_label))

        attx = torch.min(torch.max(perturbated_imgs, x-self.eps), x + self.eps)
        attx = self.norm(torch.clamp(attx, 0.0, 1.0))
        mdatt = self.source_model(attx)
        mdmat = x_match
        loss = F.smooth_l1_loss(mdatt,mdmat,reduction='none')

        loss = mask.unsqueeze(1)*loss
        loss = (loss.sum())/x.shape[0]

        return loss


