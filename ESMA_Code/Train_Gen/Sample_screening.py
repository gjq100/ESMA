import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import math

class Sample_screening:
    def __init__(self, model, train_set, device, targets, mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225], batch_size=64, q=10, normalize=True):
        self.model = model
        self.train_set = train_set
        self.device = device
        self.targets = targets
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.q = q
        self.norm = normalize

    def normalize(self, t):
        if self.norm:
            t[:, 0, :, :] = (t[:, 0, :, :] - self.mean[0]) / self.std[0]
            t[:, 1, :, :] = (t[:, 1, :, :] - self.mean[1]) / self.std[1]
            t[:, 2, :, :] = (t[:, 2, :, :] - self.mean[2]) / self.std[2]
        else:
            pass
        return t
    
    def find_thresholds(self):
        train_set = DataLoader(self.train_set, batch_size=self.batch_size, num_workers=12, shuffle=False)
        grad_norm_thr = {i: [] for i in range(len(self.targets))}
        loss_thr = {i: [] for i in range(len(self.targets))}
        criterion = nn.CrossEntropyLoss(reduction='none')

        for i, (input, label) in enumerate(train_set):
            input, label = Variable(self.normalize(input), requires_grad=True).to(self.device), label.to(self.device)
            output = self.model(input)
            loss = criterion(output, label)
            grad = torch.autograd.grad(loss, input, grad_outputs=torch.ones_like(loss), retain_graph=False,
                                       create_graph=False)[0]
            grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3))
            for j in range(label.shape[0]):
                if label[j] in self.targets:
                    index = self.targets.index(int(label[j]))
                    grad_norm_thr[index].append(grad_norm[j].item())
                    loss_thr[index].append(loss[j].item())

        for i in range(len(self.targets)):
            grad_norm_thr[i] = sorted(list(set(grad_norm_thr[i])))[self.q]
            loss_thr[i] = sorted(list(set(loss_thr[i])))[self.q]
        grad_norm_thr = np.array(list(grad_norm_thr.values()))
        loss_thr = np.array(list(loss_thr.values()))
        return grad_norm_thr, loss_thr

    def find_target_matches(self):
        target_match = {i: [] for i in range(len(self.targets))}
        grad_norm_threshold, loss_threshold = self.find_thresholds()

        i = 0
        for image, label in self.train_set.samples:
            if label in self.targets:
                index = self.targets.index(label)
                sample = self.train_set[i][0]
                criterion = nn.CrossEntropyLoss()
                input = Variable(sample, requires_grad=True).to(self.device)
                input = self.normalize(input.unsqueeze(0))
                output = self.model(input)
                sample_loss = criterion(output, torch.tensor(label).unsqueeze(0).to(self.device))
                sample_grad = torch.autograd.grad(sample_loss, input,
                                                  retain_graph=False, create_graph=False)[0]
                sample_grad_norm = torch.norm(sample_grad, p=2)
                if sample_grad_norm.item() < grad_norm_threshold[index] and sample_loss.item() < loss_threshold[index]:
                    target_match[index].append(output.detach().squeeze(0))

            i += 1

        for index in range(len(target_match)):
            target_match[index] = torch.stack(target_match[index]).mean(dim=0)
        return target_match
