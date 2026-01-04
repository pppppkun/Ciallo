import torch
from collections import defaultdict
from toolbox import *
from torch_geometric.data import Dataset, Data, Batch
from pathlib import Path
from tqdm import tqdm
from random import shuffle
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class ContinueDataset(Dataset):
    def __init__(self, datasets, pfmark=None):
        super().__init__()
        # super().__init__(root, transform, pre_transform, pre_filter, log)
        self.data = []
        self.original_graphs = []
        for dataset in datasets:
            if len(dataset[0]) == 5:
                for index, cp, original_graph, data_graph, is_pass in dataset:
                    if not is_pass:
                        self.data.append(data_graph)
                        # self.original_graphs.append(original_graph) 
            if len(dataset[0]) == 2:
                for data_graph, _ in dataset:
                    self.data.append(data_graph)
        if pfmark is None:
            self.pfmark = [0] * len(self.data)
        else:
            self.pfmark = pfmark

    def len(self):
        return len(self.data)
    
    def get(self, idx):
        return self.data[idx], self.pfmark[idx]
    

if __name__ == '__main__':
    # path = Path('dataset/binary_graph')
    costs = []
    # import os
    # for p in tqdm(list(path.iterdir())):
    #     # s = time.time()
    #     # torch.load(p)
    #     # e = time.time()
    #     # costs.append([p, e-s])
    #     # print(p, os.path.getsize(p))
    #     costs.append([p, os.path.getsize(p)])
    # # print(max(costs))
    # costs = sorted(costs, reverse=True, key=lambda x: x[1])
    # print(costs[:100])
    # torch.save(costs[:3000], 'cache_graph.list')