from random import random
import torch
import torch.nn.functional as F
import numpy as np
from toolbox import AverageMeter, ACC, multi_ACC, accuracy
from tqdm import tqdm
from torch import nn
from torch.nn import CrossEntropyLoss
from augmentation import edge_augmentation, augmentation_unlabeled, augmentation_based_on_attention
from model import AdapGAT, LogClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from ewc import EWC


class ContinueTrainer:
    def __init__(self, dataloader, cfg) -> None:
        self.dataloader = dataloader
        self.cfg = cfg
        self.ewc = None
        
    def set_ewc(self, ewc):
        """
        Set the EWC object to use for regularization.
        
        Args:
            ewc: An EWC object
        """
        self.ewc = ewc
    
    def _reset_stats(self):
        self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()
        self.ewc_loss_meter = AverageMeter()
        self.total_loss_meter = AverageMeter()

    def train(self, epoch, model, optimizer):
        self._reset_stats()
        model.train()
        device = next(model.parameters()).device
        for i, data in enumerate(self.dataloader):
            data, is_passes = data[0], data[1]
            data.to(device)
            node_feature = model(data.x, data.edge_index, data.batch, data.mask)
            graph_ids = torch.unique(data.batch)
            task_loss = 0
            
            # replay 
            for gid in graph_ids:
                nodes_in_target_graph = (data.batch == gid).nonzero(as_tuple=True)[0]
                feature = node_feature[nodes_in_target_graph]
                is_pass = is_passes[gid]
                pt = feature.clamp(min=1e-10, max=1)
                lambda_ = (1 - pt)**2
                task_loss += torch.sum(lambda_ * -torch.log(feature.clamp(min=1e-10, max=1)) * data.y[nodes_in_target_graph])
                acc = torch.argmax(feature) == torch.argmax(data.y[nodes_in_target_graph])
                self.acc_meters.update(acc.item(), 1)
            
            self.loss_meters.update(task_loss.item(), len(graph_ids))
            
            # Add EWC penalty if available
            if self.ewc is not None:
                ewc_loss = self.ewc.ewc_loss()
                self.ewc_loss_meter.update(ewc_loss.item(), 1)
                total_loss = task_loss + ewc_loss
            else:
                ewc_loss = 0
                total_loss = task_loss
                
            self.total_loss_meter.update(total_loss.item(), len(graph_ids))
            
            # Backprop and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
        payload = {
            'train/acc': self.acc_meters.avg,
            'train/task_loss': self.loss_meters.avg,
            'train/ewc_loss': self.ewc_loss_meter.avg if self.ewc else 0,
            'train/total_loss': self.total_loss_meter.avg,
        }
        return payload

    def create_ewc(self, model, importance=1000):
        """
        Create an EWC object after training on a task.
        
        Args:
            model: The PyTorch model
            prev_dataloader: DataLoader for the previous task
            importance: Importance factor for EWC penalty (lambda)
            
        Returns:
            An EWC object
        """
        return EWC(model, self.dataloader, importance)
