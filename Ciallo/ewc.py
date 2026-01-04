import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm

class EWC:
    """
    Elastic Weight Consolidation (EWC) implementation for continual learning.
    Prevents catastrophic forgetting by adding a regularization term to the loss.
    """
    def __init__(self, model, dataloader, importance=1000):
        """
        Initialize EWC.
        
        Args:
            model: The PyTorch model
            dataloader: DataLoader containing the data from the previous task
            importance: Importance factor for EWC penalty (lambda)
        """
        self.model = model
        self.dataloader = dataloader
        self.importance = importance
        self.device = next(model.parameters()).device
        
        # Store a copy of the model parameters after training on the previous task
        self.params = {n: p.clone().detach() for n, p in model.named_parameters()}
        
        # Initialize fisher information matrix
        self.fisher = self._calculate_fisher()
        
    def _calculate_fisher(self):
        """
        Calculate Fisher Information Matrix for the model parameters.
        """
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.train()
        
        # Accumulate Fisher Information
        for data, is_pass in tqdm(self.dataloader, desc="Computing Fisher Matrix"):
            data.to(self.device)
            self.model.zero_grad()
            
            # Forward pass
            node_feature = self.model(data.x, data.edge_index, data.batch, data.mask)
            
            # Calculate log likelihood for each graph
            graph_ids = torch.unique(data.batch)
            log_likelihood = 0
            for gid in graph_ids:
                nodes_in_target_graph = (data.batch == gid).nonzero(as_tuple=True)[0]
                feature = node_feature[nodes_in_target_graph]
                # Get target node based on data.y
                target_idx = torch.argmax(data.y[nodes_in_target_graph])
                # Compute log likelihood for the target node
                log_likelihood += torch.log(feature[target_idx].clamp(min=1e-10, max=1))
            
            # Backward pass to get gradients
            log_likelihood.backward()
            
            # Accumulate squared gradients
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        
        # Normalize by the number of samples
        for n in fisher.keys():
            fisher[n] = fisher[n] / len(self.dataloader)
            
        return fisher
    
    def ewc_loss(self):
        """
        Calculate EWC penalty loss.
        
        Returns:
            EWC penalty loss
        """
        loss = 0
        for n, p in self.model.named_parameters():
            # Skip parameters without Fisher information (e.g., BatchNorm statistics)
            if n in self.fisher and n in self.params:
                # Calculate EWC penalty: sum(fisher * (current_param - old_param)^2)
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
                
        return 0.5 * self.importance * loss