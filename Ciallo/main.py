import torch
import argparse
import json
import sys as _sys
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from pathlib import Path
from datetime import datetime
from toolbox import setup_seed
from dataset import ContinueDataset
from continue_cluster import exemplar_selection
from constant import dataset_path
from model import GraphConvClassifier
from trainer import ContinueTrainer
from evaluator import Metric, ContinueEvaluator
from tqdm import tqdm
from itertools import product


def parse_option():
    parser = argparse.ArgumentParser("argument for training")
    # training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--distributed", action='store_true', default=False)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--local-rank", default=-1, type=int, dest='local_rank')
    ## training for pretrained
    parser.add_argument("--t1", type=int, default=30)
    parser.add_argument("--t2", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", type=str, default="contrastive")
    # dataloader
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dataset", type=str, help="root of dataset")
    # optimizer
    parser.add_argument("--optim", type=str, default="adam", help="type of optimizer")
    parser.add_argument(
        "--lr", "--learning_rate", type=float, default=1e-5, dest="learning_rate"
    )
    parser.add_argument(
        "--wd", "--weight_decay", type=float, default=1e-4, dest="weight_decay"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    parser.add_argument("--cosine", action="store_true", default=True)
    parser.add_argument("--Tmax", type=int, default=100)
    # model
    parser.add_argument("--conv_layer", type=str, default="ggnn")
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--savepath", type=str)
    parser.add_argument(
        "--loadpath",
        type=str,
        default="runs/11_08T11_56_S_e23_bs8_lr0.003_wd0.1_mom0.9_cosTrue_tmax100_metaset.pt/model.zip",
    )
    # for GAT
    parser.add_argument("--head", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--centrality", type=str, default='degree')
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--drop_threshold", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--loss_batch", type=int)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--ema_decay', default=0.999)
    parser.add_argument('--multi', default=False, action='store_true')
    parser.add_argument('--lambda_u', type=float, default=1e-2)
    parser.add_argument('--round', type=int, default=5)
    parser.add_argument('--init_budget', type=float, default=0.1)
    parser.add_argument('--budget', type=float, default=0.15)
    parser.add_argument('--trainer', type=str, default='our')
    parser.add_argument('--label_rate', type=int, default=8)
    parser.add_argument('--ablation', type=int, default=0)
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--labeled_dataset', type=str)
    parser.add_argument('--unlabeled_dataset', type=str)
    parser.add_argument('--gpu', type=bool, default=False)
    args = parser.parse_args()
    TIMESTAMP = "{0:%m_%dT%H_%M_%S}".format(datetime.now())
    args.timestamp = TIMESTAMP
    if args.mode == "contrastive":
        args.epoch = args.t1 + args.t2
    if args.mode == 'semisupervised':
        args.T = 1.0
    if args.mode == 'active':
        args.current_round = 0
    args.command_line = _sys.argv[1:]
    return args


def init_distributed_mode(args):
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    device_ids = list(range(n_gpu))
    args.device_ids = device_ids
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    args.world_size = dist.get_world_size()


def init_optimizer(arg, model):
    if arg.optim == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=arg.learning_rate,
            weight_decay=arg.weight_decay,
            momentum=arg.momentum,
        )
    if arg.optim == "adam":
        optimizer = Adam(
            model.parameters(), lr=arg.learning_rate, weight_decay=arg.weight_decay
        )
    return optimizer


def record_hypherparams(arg):
    if arg.mode[0].upper() == 'C':
        return f"t1{arg.t1}_t2{arg.t2}_bs{arg.batch_size}_lr{arg.learning_rate}_wd{arg.weight_decay}_mom{arg.momentum}_cos{arg.cosine}_tmax{arg.Tmax}_{arg.dataset}"
    elif 'semi' in arg.mode[0]:
        return f"e{arg.epochs}_bs{arg.batch_size}_lr{arg.learning_rate}_optim{arg.optim}_ema{arg.ema}_{arg.dataset}"
    else:
        return f"e{arg.epochs}_bs{arg.batch_size}_lr{arg.learning_rate}_wd{arg.weight_decay}_mom{arg.momentum}_cos{arg.cosine}_tmax{arg.Tmax}_{arg.dataset}"


def log_dir(arg):
    flag = arg.mode[0].upper()
    if 'semi' in arg.mode:
        flag = 'SE'
    if arg.mode == 'logcla':
        flag = 'LC'
    if arg.mode == 'active':
        flag = f'AL{arg.current_round}'        
    return Path("training") / (arg.timestamp + f"_{flag}_" + record_hypherparams(arg))


def log_continue_dir(args):
    comment = args.comment + '_' if args.comment else ""
    parent_folder = Path('continue_exp') / (comment + args.timestamp + f'__lr{args.learning_rate}_bs{args.batch_size}_var{args.ablation}')
    child_folder_construct = lambda x: parent_folder / f"{x}_task"
    return parent_folder, child_folder_construct


def save_args(args):
    if 'continue' in args.mode or 'baseline' in args.mode:
        path, _ = log_continue_dir(args)
        path = (path / 'hyperparams.json').absolute()
    else:
        path = (log_dir(args) / "hyperparams.json").absolute()
    if args.distributed:
        if dist.get_rank() == 0:
            with open(path, 'w+', encoding='utf-8') as f:
                json.dump(
                    vars(args),
                    f,
                    ensure_ascii=True,
                    sort_keys=True,
                    indent=4,
                )
    else:
        with open(path, 'w+', encoding='utf-8') as f:
            json.dump(
                vars(args),
                f,
                ensure_ascii=True,
                sort_keys=True,
                indent=4,
            )


def split_dataset(dataset, test_ratio):
    test_size = int(test_ratio * len(dataset))
    train_size = len(dataset) - test_size
    trainset, testset = random_split(dataset, [train_size, test_size])
    return trainset, testset


def retrive_augmentation(dataset):
    if dataset == 'metaset.pt':
        augmentation = 'metaset_aug_edges.pt'
    return augmentation


def construct_cdataset(dataset, args):
    """
    dataset is list
    return continue learning dataset = [train1, train2, ...], [test1, test2, ...]
    """
    split_datasets = random_split(dataset, [0.2, 0.2, 0.2, 0.2, 0.2])
    trainsets = []
    testsets = []
    for dataset in split_datasets:
        train, test = random_split(dataset, [0.8, 0.2])
        trainsets.append(train)
        testsets.append(test)
    return trainsets, testsets


def append_dataset(datasets):
    x = []
    for dataset in datasets:
        for e in dataset:
            x.append(e)
    return x

def continue_learning_log(writer, metric: Metric, task):
    for name, value in metric.export(task):
        writer.add_scalar(name, value, task)


def write_into_file(file, string):
    with open(file, 'a+') as f:
        f.write(string)
        f.write('\n')



def construct_testset_by_index(datasets, index, args):
    """
    return torch dataset or list of torch dataset
    """
    if index == 0:
        return [ContinueDataset([datasets[0]])]
    else:
        return [ContinueDataset([datasets[i]]) for i in range(index+1)]


def obtain_evaluate_result(exemplar, trainset, model, args):
    """
    exemplar is [g1, g2, ...] or dataset
    trainset is ContinueDataset
    """
    evaluator = ContinueEvaluator([ContinueDataset([exemplar]), trainset], -1, None, args)
    psample, fsample = evaluator.get_pass_and_fail_example(model)
    return psample, fsample


def continue_learning(args):
    dataset = torch.load(dataset_path / args.dataset)
    trainsets, testsets = construct_cdataset(dataset, args)
    model = GraphConvClassifier(args)
    if args.gpu:
        model = model.to('cuda:0')
    else:
        pass
    optimizer = init_optimizer(args, model)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Tmax)
    ewc = None
    exemplar = None
    psample = None
    fsample = None
    metric = Metric()
    # writer = SummaryWriter(log_dir=(args))
    parent_folder, child_folder_construct = log_continue_dir(args)
    parent_folder.mkdir()
    writer = SummaryWriter(parent_folder)
    save_args(args)
    s = time.time()
    for task_idx in range(len(trainsets)):
        # child_folder = child_folder_construct(task_idx)
        # writer = SummaryWriter(child_folder)
        trainset = ContinueDataset([trainsets[task_idx]])
        testset = construct_testset_by_index(testsets, task_idx, args)
        evaluator = ContinueEvaluator(testset, task_idx, metric, args)
        if task_idx == 0:
            dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        else:
            exemplar = exemplar_selection(trainsets[:task_idx], args)
            hybrid_dataset = ContinueDataset([exemplar, trainsets[task_idx]])
            dataloader = DataLoader(hybrid_dataset, batch_size=args.batch_size, shuffle=True)
        trainer = ContinueTrainer(dataloader, args)
        trainer.set_ewc(ewc)
        for epoch in tqdm(range(args.epochs), desc='training the model'):
            trainer.train(epoch, model, optimizer)
            evaluator.evaluate(epoch, model)
            if args.cosine:
                scheduler.step()
        continue_learning_log(writer, metric, task_idx)
        ewc = trainer.create_ewc(model=model)
    e = time.time()
    write_into_file('CLI_baseline.txt', f'Our-time: {e-s}')


def main():
    args = parse_option()
    setup_seed(args.seed)
    if args.mode == 'continue':
        continue_learning(args)


if __name__ == "__main__":
    main()
