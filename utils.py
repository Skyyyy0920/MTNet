import os
import re
import dgl
import glob
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt


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


def create_mapping_matrix(df, POI_id2idx_dict, cat_id2idx_dict):
    n, m = len(cat_id2idx_dict), len(POI_id2idx_dict)
    mapping_matrix = np.zeros((n, m))
    for index, row in df.iterrows():
        POI_idx = POI_id2idx_dict[row['POI_id']]
        cat_idx = cat_id2idx_dict[row['POI_catid']]
        mapping_matrix[cat_idx][POI_idx] = 1
    for i in range(n):
        mapping_matrix[i] = mapping_matrix[i] / np.sum(mapping_matrix[i])
    return mapping_matrix


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


def get_performance(y_true_seq, y_pred_seq):
    acc = []
    for k in [1, 5, 10, 20]:
        acc.append(top_k_acc(y_true_seq, y_pred_seq, k))
    return acc[0], acc[1], acc[2], acc[3]


def get_pred_label(y_label_list, y_pred_list):
    y_label_POI_numpy = np.concatenate(y_label_list, axis=0)
    y_pred_POI_numpy = np.concatenate(y_pred_list, axis=0)
    # none_label = np.where(y_label_POI_numpy == -1)
    # y_label_POI_numpy = np.delete(y_label_POI_numpy, none_label)
    # y_pred_POI_numpy = np.delete(y_pred_POI_numpy, none_label)
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


def add_children(tree, trajectory, index, idx2idx_dict, flag_dict, nary):
    """
    Using DFS to construct the tree for N-ary TreeLSTM
    """
    node = trajectory[index]
    idx2idx_dict[index] = tree.number_of_nodes()
    tree.add_node(idx2idx_dict[index], x=node['features'], y=node['labels'], mask=1)

    if index > 0 and flag_dict[index]:
        flag_dict[index] = 0  # already play as parent node
        for i in range(nary, 0, -1):
            if index - i >= 0:
                add_children(tree, trajectory, index - i, idx2idx_dict, flag_dict, nary)
                tree.add_edge(idx2idx_dict[index - i], idx2idx_dict[index])  # src -> dst
            else:  # fictitious node
                node_id = tree.number_of_nodes()
                tree.add_node(node_id, x=trajectory[0]['features'], y=[-1] * 4, mask=0)
                tree.add_edge(node_id, idx2idx_dict[index])  # src -> dst
    return


def add_children_out(tree, trajectory, index, idx2idx_dict, flag_dict, nary):
    re_index = len(trajectory) - 1 - index
    max_index = re_index
    node = trajectory[re_index]
    idx2idx_dict[index] = tree.number_of_nodes()
    tree.add_node(idx2idx_dict[index], x=node['features'], y=node['labels'], mask=1)

    if index > 0 and flag_dict[index]:
        flag_dict[index] = 0  # already play as parent node
        for i in range(nary, 0, -1):
            if index - i < 0:  # fictitious node
                node_id = tree.number_of_nodes()
                tree.add_node(node_id, x=trajectory[-1]['features'], y=[-1] * 4, mask=0)
                tree.add_edge(node_id, idx2idx_dict[index])  # src -> dst
                max_index = len(trajectory) - 1
            else:
                child_idx = add_children_out(tree, trajectory, index - i, idx2idx_dict, flag_dict, nary)
                max_index = child_idx if max_index < child_idx else max_index
                tree.add_edge(idx2idx_dict[index - i], idx2idx_dict[index])  # src -> dst

    # change node label
    tree.add_node(idx2idx_dict[index], x=node['features'], y=trajectory[max_index]['labels'], mask=1)
    return max_index


def construct_dgl_tree(trajectory, cell_type, nary, need_plot, tree_type):
    tree = nx.DiGraph()
    idx2idx_dict = {}
    flag_dict = dict(zip(range(len(trajectory)), np.ones(len(trajectory))))

    start_index = len(trajectory) - 1
    if tree_type == 'in':  # in tree
        add_children(tree, trajectory, start_index, idx2idx_dict, flag_dict, nary)
    elif tree_type == 'out':  # out tree
        add_children_out(tree, trajectory, start_index, idx2idx_dict, flag_dict, nary)
    else:
        print("Tree type wrong!")

    if need_plot:
        plot_tree(tree)  # optional

    dgl_tree = dgl.from_networkx(tree, node_attrs=['x', 'y', 'mask'])
    return dgl_tree
