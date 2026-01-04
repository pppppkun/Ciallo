import re
import subprocess
import torch
import csv
import pickle
import json
import networkx as nx
import pandas as pd
import numpy as np
import argparse
from torch_geometric.utils import from_networkx
from torch_geometric.nn import Node2Vec
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from colorama import Fore, Back, Style
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from node2vec import Node2Vec as n2v
from functools import lru_cache
import matplotlib.pyplot as plt

error_line = []
device = 'cuda:1'

@dataclass
class GlobalWrapper:
    def __init__(self) -> None:
        pass
    
    def __str__(self) -> str:
        return f'{self.__dict__}'

global_wrapper = GlobalWrapper()


class LogLine:
    def __init__(self, logline):
        self.error = False
        self.file = None
        self.pt_error = False
        self.function = None
        self.module = None
        self.logline = logline
        mark = [
            "INF<e>",
            "ERR<eina_safety>",
            "INF<tvs>",
            "MSG<tv>",
            "INF<tv>",
            # "tv_stream_videosrc_resolution_info_change_cb",
        ]
        for m in mark:
            if m in logline:
                self.error = True
                return
        # if 'INF<e>'
        # match the file name
        file_pt = re.compile(r"(\.cpp:|\.c:|\.cc:|\.h:)")
        candidate_file = file_pt.findall(logline)
        if len(candidate_file) != 1:
            if len(candidate_file) == 0:
                flag, file, func = special_line(logline)
                if flag:
                    self.file = file
                    self.function = func
                else:
                    self.error = True
                    error_line.append(logline)
                    # return
            if len(candidate_file) > 1:
                candidate_file = candidate_file[:1]
        if len(candidate_file) == 1:
            tmp = candidate_file[0]
            tmp = logline.index(tmp)
            file = ""
            for i in range(tmp - 1, 0, -1):
                if logline[i] not in [
                    " ",
                    "/",
                    "[",
                    "]",
                    "(",
                    ")",
                    "!",
                    "@",
                    "#",
                    ":",
                    "<",
                    ">",
                    "{",
                    "}",
                ]:
                    file = logline[i] + file
                else:
                    break
            # file = file + candidate_file[0][:-1]
            self.file = file

            # match the function
            end_of_file = tmp + len(candidate_file[0])  # the next char after ':'
            start_of_function = None
            function = ""
            if end_of_file >= len(logline):
                error_line.append(logline)
                self.error = True
                return            
            if logline[end_of_file] in [" ", "~"]:
                start_of_function = end_of_file + 1
            if logline[end_of_file].isalpha():
                start_of_function = end_of_file
            if logline[end_of_file].isdigit():
                # pass
                for i in range(end_of_file, len(logline)):
                    if logline[i].isalpha():
                        start_of_function = i
                        break
            # else:
            #     # only process 3 case
            #     if logline[end_of_file] == ''
            if start_of_function:
                if start_of_function >= len(logline):
                    error_line.append(logline)
                    self.error = True
                    return                
                if logline[start_of_function] == "~":
                    start_of_function += 1
                for i in range(start_of_function, len(logline)):
                    if logline[i] not in [
                        " ",
                        "/",
                        "[",
                        "]",
                        "(",
                        ")",
                        "!",
                        "@",
                        "#",
                        ":",
                        "<",
                        ">",
                        "{",
                        "}",
                        ",",
                    ]:
                        function = function + logline[i]
                    else:
                        break
            self.function = function

        # must contain process id and thread id
        try:
            ptid = re.compile(r".*\(P *([0-9]*), *T *([0-9]*)\).*")
            m = ptid.match(logline)
            self.pid = m.group(1)
            self.tid = m.group(2)
            # print(self.pid, self.tid)
        except AttributeError:
            # print(logline)
            error_line.append(logline)
            self.pt_error = True
            self.error = True

        # match the log type/module name
        try:
            possible_time = re.compile(r"\[[0-9]+:[0-9]+:[0-9]+\]")
            m = possible_time.match(logline)
            # print(m)
            if m:
                s, e = m.span(0)
                logline_ = logline[e:].strip()
            else:
                logline_ = logline
            module_pt = re.compile(r"([0-9]+.[0-9]+)* *([\S]*) *\(")
            m = module_pt.match(logline_)
            self.time = m.group(1)
            self.type, self.module = m.group(2).split("/")
        except AttributeError:
            # print(logline)
            error_line.append(logline)
            self.error = True

        if self.error:
            if not self.file and self.module:
                self.file = self.module + '_f'
                self.error = False
        

        if not self.error:
            if self.pt_error:
                print(logline)


    def __str__(self):
        return f"{self.module}, {self.file}, {self.function}"

def parse_option():
    parser = argparse.ArgumentParser("argument for preprocess")
    parse_algorithm = [
        "graph",
        "language",
    ]

    parser.add_argument(
        "-s", "--single", action="store_true", help="only proces single dlog"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbose mode", default=False
    )
    parser.add_argument(
        "--save", action="store_true", help="save the preprocess data to dataset"
    )
    parser.add_argument(
        "--check", action='store_true', default=False
    )
    parser.add_argument("--output", "-o", action="store", help="output file name")
    parser.add_argument(
        "--info", "-i", action="store_true", help="show info of meta data"
    )
    parser.add_argument(
        '--alg', action='store', default='graph'
    )
    parser.add_argument(
        '--load_pretrained', action='store_true', default=False
    )
    parser.add_argument(
        '--test', action='store_true', default=False
    )
    parser.add_argument(
        '--clusters', type=int, default=70, help='number of clusters'
    )
    parser.add_argument(
        '--number', type=int, default=40
    )
    parser.add_argument(

    )
    args = parser.parse_args()
    # args.single = True
    # args.verbose = True
    # args.cov = 'function'
    return args


def parse_log_to_graph(dlog_path, graph: nx.DiGraph):
    args = global_wrapper.args
    dlog = Path(dlog_path)
    dlog = dlog.read_text(encoding="utf-8", errors="ignore").split("\n")
    dlog = list(filter(lambda x: x, map(lambda x: x.strip(), dlog)))[3:]
    hierarchy = defaultdict(lambda: defaultdict(list))
    function_set = defaultdict(lambda: defaultdict(set))
    for i, line in enumerate(dlog):
        # try:
        logobj = LogLine(line)
        # except:
        #     print(dlog_path, i, line)
        if not logobj.error:
            if logobj.function not in function_set[logobj.module][logobj.file]:
                hierarchy[logobj.module][logobj.file].append(logobj.function)
                function_set[logobj.module][logobj.file].add(logobj.function)
    
    # insert log node into graph
    dlog_name = Path(dlog_path).stem
    for module in hierarchy:
        graph.add_edge(dlog_name, module)
        for file in hierarchy[module]:
            graph.add_edge(module, file)
            for function in hierarchy[module][file]:
                graph.add_edge(file, function)
    if args.verbose:
        print(graph)


def get_embedding_for_node(model, node):
    node_tables = global_wrapper.node_tables
    index = node[node_tables]
    embedding = model(index)
    return embedding


def get_embedding_for_graph(model, graph: nx.DiGraph):
    # embeddings = list(map(lambda x: get_embedding_for_node(model, x[1]), enumerate(graph.nodes)))
    node_tables = global_wrapper.node_tables
    indexs = list(map(lambda x: node_tables[x[1]], enumerate(graph.nodes)))
    embeddings = model(torch.tensor(indexs))
    graph_embed = torch.sum(embeddings, dim=0)
    return graph_embed


def get_embedding_for_graph_v2(model, graph):
    # indexs = list(map(lambda x: model.key_to_index[x[1]['identifier']], enumerate(graph.nodes)))
    # embeddings = model(torch.tensor(indexs))
    indexs = []
    for n in graph.nodes:
        index = model.key_to_index[graph.nodes[n]['identifier']]
        indexs.append(index)
    embeddings = model.vectors[torch.tensor(indexs)]
    graph_embed = np.mean(embeddings, axis=0)
    return graph_embed


def get_graph_list(trainsets):
    gl = []
    for trainset in trainsets:
        # for _, _, g, _, is_pass in trainset:
        #     if not is_pass:
        #         gl.append(g)
        for a,b,c,d,is_pass in trainset:
            gl.append([a,b,c,d,is_pass])
    return gl


def related_to_fault(graph, src, dst):
    return graph.nodes[src]['y'] == 1 or graph.nodes[dst]['y'] == 1

def exemplar_selection(trainsets, args):
    uni_graph = nx.DiGraph()
    graph_list = get_graph_list(trainsets)
    for _, _, graph, _, is_pass in graph_list:
        for src, dst, edge_attrs in graph.edges(data=True):
            n1 = graph.nodes[src]['identifier']
            n2 = graph.nodes[dst]['identifier']
            if not uni_graph.has_edge(n1, n2):
                edge_attrs['weight'] = 1
                if related_to_fault(graph, src, dst):
                    edge_attrs['fault_degree'] = 1
                uni_graph.add_edge(n1, n2)
                uni_graph[n1][n2].update(edge_attrs)
            else:
                edge_attrs = uni_graph[n1][n2]
                edge_attrs['weight'] += 1
                if related_to_fault(graph, src, dst):
                    if 'fault_degree' not in edge_attrs:
                        edge_attrs['fault_degree'] = 1
                    else:
                        edge_attrs['fault_degree'] += 1
                uni_graph[n1][n2].update(edge_attrs)

    for src, dst, edge_attrs in uni_graph.edges(data=True):
        a = 0
        if 'fault_degree' in edge_attrs:
            a = edge_attrs['fault_degree']
        edge_attrs['weight'] = np.log(edge_attrs['weight'] + a)

    node2vec = n2v(uni_graph, dimensions=128, walk_length=20, num_walks=10)
    model = node2vec.fit(window=10, min_count=1, batch_words=4).wv

    graph_embeds = []
    identifiers = []
    for index, data in enumerate(graph_list):
        _, _, graph, _, is_pass = data
        embedding = get_embedding_for_graph_v2(model, graph)
        graph_embeds.append(embedding)
        identifiers.append(index)
    
    graph_embeds = np.array(graph_embeds)
    identifiers = np.array(identifiers)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels = dbscan.fit_predict(graph_embeds)
    K = args.number
    selected_indices = []
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]

    print(f"Found {len(unique_clusters)} clusters ")

    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) <= K:
            tmp_selected_indices = cluster_indices
        else:
            tmp_selected_indices = np.random.choice(cluster_indices, K, replace=False)

        selected_indices.extend(identifiers[tmp_selected_indices])
        print(f'Cluster {cluster_id}: selected {len(tmp_selected_indices)} samples from {len(cluster_indices)} total')

    exemplar = []
    for index in selected_indices:
        exemplar.append(graph_list[index])
    return exemplar


def visual(data, km: KMeans):
    print('viaual data')
    labels = km.labels_
    unique_labels = set(labels)
    tx = data[:, 0]
    ty = data[:, 1]
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.scatter(tx, ty, c='black')
    centroids = km.cluster_centers_
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="x", c='red', linewidths=3, s=169, zorder=10)
    plt.savefig('unlabeled_logs1.png')
    plt.close()


def check(df):
    print('-'*20 + 'check dlog path whether exists.' + '-'*20)
    count = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        fail_file = row['fail_dlog_file']
        pass_file = row['pass_dlog_file']
        fail_file = fail_file.replace('\\', '/')
        pass_file = pass_file.replace('\\', '/')
        fail_file = Path(fail_file)
        pass_file = Path(pass_file)
        if not fail_file.exists():
            count += 1
    print(count)
