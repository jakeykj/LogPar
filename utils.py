import os
from copy import deepcopy

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class EarlyStopping(object):
    """Stops the training when the improvement on validation set 
       does not improve for a certain number of epochs.
    """
    def __init__(self, patience=7, score_tol=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None  # the higher the better
        self.early_stopped = False
        self.score_tol = score_tol
        self.best_model = None

    def __call__(self, val_score, model):
        if self.best_score is None or val_score > self.best_score + self.score_tol:
            self.best_score = val_score
            self.best_model = deepcopy(model).cpu()
            self.counter = 0
        else:
            self.counter += 1
            self.early_stopped = self.counter >= self.patience
        return self.early_stopped


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PaddedDenseTensor:
    def __init__(self, temporal_input, num_feats, subset='train'):
        self.num_feats = num_feats
        self.num_visits = [len(pt['times']) for pt in temporal_input]
        self.max_visits = max(self.num_visits)
        self.subset = subset

        self.Xs = [torch.LongTensor(pt[subset][:, :-1]) if pt[subset].size > 0 else torch.LongTensor() 
                   for pt in temporal_input]
        self.Xvals = [torch.FloatTensor(pt[subset][:, -1]) if pt[subset].size > 0 else torch.FloatTensor() 
                      for pt in temporal_input]

        # Xs = [torch.LongTensor(pt[subset]) if pt[subset].size > 0 else torch.LongTensor() 
        #       for pt in temporal_input]
        # Xsparse = [torch.cat([i*torch.ones(X.shape[0], 1).long(), X], dim=1) 
        #            if X.numel() != 0 else torch.LongTensor()
        #            for i, X in enumerate(Xs)]
        # Xsparse = torch.cat(Xsparse, dim=0)
        # self.Xdense = torch.zeros(len(temporal_input), self.max_visits, self.num_feats)
        # self.Xdense[Xsparse[:, 0], Xsparse[:, 1], Xsparse[:, 2]] = Xsparse[:, 3].float()
        # if subset == 'train':
        #     self.masks = torch.cat([(self.num_visits[p] > torch.arange(self.max_visits)).float().unsqueeze(0) 
        #                        for p in range(len(temporal_input))], dim=0).unsqueeze(2)
        # else:
        #     self.masks = torch.zeros_like(self.Xdense)
        #     self.masks[Xsparse[:,0], Xsparse[:, 1], Xsparse[:, 2]] = 1
        
        
        self.hf_labels = [pt['label'] for pt in temporal_input]
        self.times = [pt['times'] for pt in temporal_input]
        self.deltas = [pt['deltas'] for pt in temporal_input]


    def __call__(self, pids):
        pids = torch.LongTensor([pid[0].item() for pid in pids])

        Xtmp = [torch.cat([i*torch.ones(self.Xs[pid].shape[0], 1).long(), self.Xs[pid]], dim=1) 
                if self.Xs[pid].numel() != 0 else torch.LongTensor()
                for i, pid in enumerate(pids)]
        Xsparse = torch.cat(Xtmp, dim=0)
        
        Xdense = torch.zeros(len(pids), self.max_visits, self.num_feats)
        Xdense[Xsparse[:,0], Xsparse[:, 1], Xsparse[:, 2]] = torch.cat([self.Xvals[pid] for pid in pids], axis=0)

        if self.subset == 'train':  # padding mask for training
            masks = torch.cat([(self.num_visits[pid] > torch.arange(self.max_visits)).float().unsqueeze(0) 
                               for pid in pids], dim=0).unsqueeze(2)
        else:
            masks = torch.zeros_like(Xdense)
            masks[Xsparse[:,0], Xsparse[:, 1], Xsparse[:, 2]] = 1

        deltas = [torch.FloatTensor(self.deltas[pid] + [0] * (self.max_visits-len(self.deltas[pid]))).unsqueeze(0)
                  for pid in pids]
        deltas = torch.cat(deltas, dim=0)

        # Xdense = self.Xdense[pids]
        # masks = self.masks[pids]
        return pids, Xdense, masks, deltas