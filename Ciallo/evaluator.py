from toolbox import AverageMeter, first_rank, ACC, multi_ACC, average_rank, accuracy
from itertools import product
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from collections import defaultdict
# if torch.cuda.is_available():
#     device = torch.device("cuda:1")
# else:
#     device = torch.device("cpu")


class MetricInTask:
    def __init__(self):
        self.acc1 = 0
        self.acc3 = 0
        self.acc5 = 0
        self.firstrank = 0
        self.reciprocalrank = 0
        self.mar = 0

class Metric:
    def __init__(self):
        self.data = defaultdict(lambda: defaultdict(MetricInTask))
        self.dimension = ('acc1', 'acc3', 'acc5', 'firstrank', 'mar')
        self.function = {
            'aa': self.aa,
            'aia': self.aia,
            'fm': self.fm,
            'bwt': self.bwt
        }


    def export(self, task):
        for a, b in product(self.dimension, self.function.keys()):
            f = self.function[b]
            v = f(task, a)
            yield f'{a}/{b}', v
            yield f'{b}/{a}', v


    def get_value_of_dimension(self, task, dimension):
        var = self.data[task]
        result = []
        for i in range(task + 1):
            if dimension == 'acc1':
                result.append(var[i].acc1)
            elif dimension == 'acc3':
                result.append(var[i].acc3)
            elif dimension == 'acc5':
                result.append(var[i].acc5)
            elif dimension == 'firstrank':
                result.append(var[i].firstrank)
            elif dimension == 'reciprocalrank':
                result.append(var[i].reciprocalrank)
            elif dimension == 'mar':
                result.append(var[i].mar)
        return result


    def aa(self, task, dimension):
        res = self.get_value_of_dimension(task, dimension)
        return sum(res) / len(res) if res else 0

    def aia(self, task, dimension):
        res = []
        for i in range(task + 1):
            value = self.aa(i, dimension)
            res.append(value)
        return sum(res) / len(res) if res else 0

    def fm(self, task, dimension):
        if task == 0:
            return 0
        result = []
        for i in range(task): 
            final = self.get_value_of_dimension(task, dimension)[i]
            max_performance = -float('inf')
            for j in range(i, task + 1):
                current_perf = self.get_value_of_dimension(j, dimension)[i]
                max_performance = max(max_performance, current_perf)
            result.append(max_performance - final)
        return sum(result) / len(result) if result else 0

    def bwt(self, task, dimension):
        result = []
        if task == 0:
            return 0
        for i in range(task):
            v = self.get_value_of_dimension(i, dimension)[i]
            r = self.get_value_of_dimension(task, dimension)[i]
            result.append(r - v)
        return sum(result) / len(result) if result else 0


class ContinueEvaluator:
    def __init__(self, testsets, current_task, metric: Metric, args):
        """
        testsets = [test1, test2, test3, ...]
        test_i is ContinueDataset
        """
        self.args = args
        # self.dataloader = dataloader
        self.testsets = testsets
        self.current_task = current_task
        self.criterion = CrossEntropyLoss()
        self.metric = metric
        self.acc_f = multi_ACC

    def _reset_stats_atom(self):
        self.acc1_meters = AverageMeter()
        self.acc3_meters = AverageMeter()
        self.acc5_meters = AverageMeter()
        self.loss_meters = AverageMeter()
        self.firstrank_meters = AverageMeter()
        self.reciprocalrank_meters = AverageMeter()
        self.mar = AverageMeter()

    def _reset_stats(self):
        self.aa = AverageMeter()
        self.aia = AverageMeter()
        self.fm = AverageMeter()
        self.bwt = AverageMeter()
        self.im = AverageMeter()
        self.fwt = AverageMeter()

    
    def update_metric(self, task):
        self.metric.data[self.current_task][task].acc1 = self.acc1_meters.avg
        self.metric.data[self.current_task][task].acc3 = self.acc3_meters.avg
        self.metric.data[self.current_task][task].acc5 = self.acc5_meters.avg
        self.metric.data[self.current_task][task].firstrank = self.firstrank_meters.avg
        self.metric.data[self.current_task][task].reciprocalrank = self.reciprocalrank_meters.avg
        self.metric.data[self.current_task][task].mar = self.mar.avg
        

    def evaluate(self, epoch, model):
        model.eval()
        device = next(model.parameters()).device
        for index, testset in enumerate(self.testsets):
            testloader = DataLoader(testset, batch_size=self.args.batch_size, shuffle=False)
            self._reset_stats_atom()
            for i, data in enumerate(testloader):
                data, _ = data[0], data[1]
                data.to(device)
                with torch.no_grad():
                    node_feature = model(data.x, data.edge_index, data.batch, data.mask)
                    graph_ids = torch.unique(data.batch)
                    loss = 0
                    for gid in graph_ids:
                        nodes_in_target_graph = (data.batch == gid).nonzero(as_tuple=True)[
                            0
                        ]
                        feature = node_feature[nodes_in_target_graph]
                        y = data.y[nodes_in_target_graph]
                        loss += torch.sum(
                            -torch.log(feature.clamp(min=1e-10, max=1)) * y, dim=-1
                        )
                        acc = self.acc_f(feature, y, [1, 3, 5])
                        _first_rank = first_rank(feature, y)[0]
                        _average_rank = average_rank(feature, y)
                        self.acc1_meters.update(acc[0], 1)
                        self.acc3_meters.update(acc[1], 1)
                        self.acc5_meters.update(acc[2], 1)
                        self.firstrank_meters.update(_first_rank, 1)
                        self.reciprocalrank_meters.update(1 / _first_rank, 1)
                        self.mar.update(_average_rank, 1)
            if epoch == self.args.epochs - 1:
                self.update_metric(index)



    def get_pass_and_fail_example(self, model):
        exemplar, trainset = self.testsets[0], self.testsets[1]
        loader1 = DataLoader(exemplar, batch_size=1, shuffle=False)
        loader2 = DataLoader(trainset, batch_size=1, shuffle=False)
        device = next(model.parameters()).device
        model.eval()
        pass_graph, error_graph = [], []
        for i, data in enumerate(loader1):
            data, _ = data
            data.to(device)
            with torch.no_grad():
                node_feature = model(data.x, data.edge_index, data.batch, data.mask)
                graph_ids = torch.unique(data.batch)
                for gid in graph_ids:
                    nodes_in_target_graph = (data.batch == gid).nonzero(as_tuple=True)[0]
                    feature = node_feature[nodes_in_target_graph]
                    y = data.y[nodes_in_target_graph]
                    acc = self.acc_f(feature, y, [1, 3, 5])
                    if acc[0] == 1:
                        pass_graph.append(exemplar[i])
                    else:
                        error_graph.append(exemplar[i])
        for i, data in enumerate(loader2):
            data, _ = data
            data.to(device)
            with torch.no_grad():
                node_feature = model(data.x, data.edge_index, data.batch, data.mask)
                graph_ids = torch.unique(data.batch)
                for gid in graph_ids:
                    nodes_in_target_graph = (data.batch == gid).nonzero(as_tuple=True)[0]
                    feature = node_feature[nodes_in_target_graph]
                    y = data.y[nodes_in_target_graph]
                    acc = self.acc_f(feature, y, [1, 3, 5])
                    if acc[0] == 1:
                        pass_graph.append(trainset[i])
                    else:
                        error_graph.append(trainset[i])
        return pass_graph, error_graph