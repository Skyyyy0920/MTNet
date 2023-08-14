import os
import re
import dgl
import glob
import torch
import random
import collections
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SSTBatch = collections.namedtuple(
    "SSTBatch", ["graph", "features", "time", "label", "mask", "mask2", "type"]
)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zipdir(path, zipf, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                zipf.write(filename, arcname)


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def top_k_acc(y_true_seq, y_pred_seq, k):
    hit = 0
    count = 0
    # Convert to binary relevance (nonzero is relevant).
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        if y_true == -1:
            continue
        top_k_rec = y_pred.argsort()[-k:][::-1]
        idx = np.where(top_k_rec == y_true)[0]
        if len(idx) != 0:
            hit += 1
        count += 1
    return hit / count


def MRR_metric(y_true_seq, y_pred_seq):
    rlt = 0
    count = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        if y_true == -1:
            continue
        rec_list = y_pred.argsort()[-len(y_pred):][::-1]
        r_idx = np.where(rec_list == y_true)[0][0]
        rlt += 1 / (r_idx + 1)
        count += 1
    return rlt / count


def get_performance(y_true_seq, y_pred_seq):
    acc = []
    for k in [1, 5, 10, 20]:
        acc.append(top_k_acc(y_true_seq, y_pred_seq, k))
    mrr = MRR_metric(y_true_seq, y_pred_seq)
    return acc[0], acc[1], acc[2], acc[3], mrr


def get_pred_label(y_label_list, y_pred_list):
    y_label_POI_numpy = np.concatenate(y_label_list, axis=0)
    y_pred_POI_numpy = np.concatenate(y_pred_list, axis=0)
    return y_label_POI_numpy, y_pred_POI_numpy


def process_for_GowallaCA(df):
    pd.options.mode.chained_assignment = None
    df.insert(loc=2, column='POI_catid', value='')
    df.insert(loc=7, column='timezone', value=0)
    df.insert(loc=8, column='UTC_time', value='')
    df.insert(loc=10, column='day_of_week', value=0)
    df = df[df['POI_catname'] != 'dummy']
    df.rename(columns={'checkin_time': 'local_time'}, inplace=True)
    df['POI_catid'] = df.apply(lambda x: eval(x['POI_catname'].replace(";", ","))[0]['url'], axis=1)
    return df


def plot_tree(g):
    # this plot requires pygraphviz package
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    nx.draw_networkx(g,
                     pos,
                     with_labels=False,
                     node_size=20,
                     node_color=[[0.5, 0.5, 0.5]],
                     arrowsize=8)
    node_labels = nx.get_node_attributes(g, 'x')
    nx.draw_networkx_labels(g, pos, labels=node_labels, font_color='blue')
    node_labels = nx.get_node_attributes(g, 'y')
    nx.draw_networkx_labels(g, pos, labels=node_labels, font_color='red')
    node_labels = nx.get_node_attributes(g, 'mask')
    nx.draw_networkx_labels(g, pos, labels=node_labels, font_color='green')
    plt.show()


def add_true_node(tree, trajectory, index, parent_node_id, nary):
    for i in range(nary - 1, 0, -1):
        if index - i >= 0:
            node_id = tree.number_of_nodes()
            node = trajectory[index - i]
            tree.add_node(node_id, x=node['features'], time=node['time'], y=node['labels'], mask=1, mask2=0, type=2)
            tree.add_edge(node_id, parent_node_id)
        else:  # empty node
            node_id = tree.number_of_nodes()
            tree.add_node(node_id, x=[0] * 4, time=0, y=[-1] * 3, mask=0, mask2=0, type=-1)
            tree.add_edge(node_id, parent_node_id)

    sub_parent_node_id = tree.number_of_nodes()
    tree.add_node(sub_parent_node_id, x=[0] * 4, time=0, y=[-1] * 3, mask=0, mask2=0, type=-1)
    tree.add_edge(sub_parent_node_id, parent_node_id)

    if index - (nary - 1) > 0:
        add_true_node(tree, trajectory, index - (nary - 1), sub_parent_node_id, nary)
        tree.add_node(sub_parent_node_id, x=[0] * 4, time=0, y=trajectory[index - (nary - 1)]['labels'], mask=0,
                      mask2=0, type=-1)


def add_period_node(tree, trajectory, nary):
    node_id = tree.number_of_nodes()
    period_label = trajectory[len(trajectory) - 1]['labels'] if len(trajectory) > 0 else [-1] * 3
    tree.add_node(node_id, x=[0] * 4, time=0, y=period_label, mask=0, mask2=1, type=1)

    if len(trajectory) > 0:
        add_true_node(tree, trajectory, len(trajectory), node_id, nary)

    return node_id


def add_day_node(tree, trajectory, labels, index, nary):
    node_id = tree.number_of_nodes()
    tree.add_node(node_id, x=[0] * 4, time=0, y=labels[index], mask=0, mask2=1, type=0)
    if index > 0:  # recursion
        child_node_id = add_day_node(tree, trajectory, labels, index - 1, nary)
        tree.add_edge(child_node_id, node_id)
    else:
        fake_node_id = tree.number_of_nodes()
        tree.add_node(fake_node_id, x=[0] * 4, time=0, y=[-1] * 3, mask=0, mask2=0, type=-1)
        tree.add_edge(fake_node_id, node_id)

    day_trajectory = trajectory[index]
    for i in range(len(day_trajectory)):  # Four time periods， 0-6， 6-12， 12-18， 18-24
        period_node_id = add_period_node(tree, day_trajectory[i], nary)
        tree.add_edge(period_node_id, node_id)

    return node_id


def construct_MobilityTree(trajectory, labels, nary, need_plot):
    tree = nx.DiGraph()
    add_day_node(tree, trajectory, labels, len(trajectory) - 1, nary)

    if need_plot:
        plot_tree(tree)  # optional

    dgl_tree = dgl.from_networkx(tree, node_attrs=['x', 'time', 'y', 'mask', 'mask2', 'type'])
    return dgl_tree
