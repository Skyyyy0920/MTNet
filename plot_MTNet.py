import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataset import *
from sklearn.cluster import KMeans


def plot_tree(g):
    # this plot requires pygraphviz package
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    nx.draw_networkx(g,
                     pos,
                     with_labels=False,
                     node_size=20,
                     node_color=[[0.5, 0.5, 0.5]],
                     arrowsize=8)
    # node_labels = nx.get_node_attributes(g, 'x')
    # nx.draw_networkx_labels(g, pos, labels=node_labels, font_color='blue')
    node_labels = nx.get_node_attributes(g, 'y')
    nx.draw_networkx_labels(g, pos, labels=node_labels, font_color='red')
    # node_labels = nx.get_node_attributes(g, 'time')
    # nx.draw_networkx_labels(g, pos, labels=node_labels, font_color='green')
    plt.show()


def add_children(tree, trajectory, index, idx2idx_dict, flag_dict, nary):
    """
    Using DFS to construct the tree for N-ary TreeLSTM
    """
    node = trajectory[index]
    idx2idx_dict[index] = tree.number_of_nodes()
    if index > 0:
        tree.add_node(idx2idx_dict[index], u=node['features'][0], x=node['features'][1], time=node['time'],
                      y=node['labels'], mask=1, type=0)  # add parent node
    else:
        tree.add_node(idx2idx_dict[index], u=node['features'][0], x=node['features'][1], time=node['time'],
                      y=node['labels'], mask=1, type=5)  # add parent node

    if index > 0 and flag_dict[index] > 0:
        flag_dict[index] -= 1  # already play as parent node
        for i in range(nary, 0, -1):
            if index - i >= 0:
                add_children(tree, trajectory, index - i, idx2idx_dict, flag_dict, nary)
                tree.add_edge(idx2idx_dict[index - i], idx2idx_dict[index])  # src -> dst
            else:  # fictitious node
                node_id = tree.number_of_nodes()
                tree.add_node(node_id, u=node['features'][0], x=trajectory[0]['features'][1], time=node['time'],
                              y=[-1] * 3, mask=0, type=-1)
                tree.add_edge(node_id, idx2idx_dict[index])  # src -> dst

        node_id = tree.number_of_nodes()
        tree.add_node(node_id, u=node['features'][0], x=node['features'][1], time=node['time'],
                      y=node['labels'], mask=1, type=3)  # add parent node
        tree.add_edge(node_id, idx2idx_dict[index])  # src -> dst
        for i in range(1, 3):
            node_id = tree.number_of_nodes()
            tree.add_node(node_id, u=node['features'][0], x=node['features'][i + 1], time=node['time'],
                          y=node['labels'], mask=1, type=i)  # 1 POI, 2 cat, 3 coo
            tree.add_edge(node_id, idx2idx_dict[index])  # src -> dst

    return


def add_children_out(tree, trajectory, index, idx2idx_dict, flag_dict, nary):
    re_index = len(trajectory) - 1 - index
    max_index = re_index
    node = trajectory[re_index]
    idx2idx_dict[index] = tree.number_of_nodes()
    if index > 0:
        tree.add_node(idx2idx_dict[index], u=node['features'][0], x=node['features'][1], time=node['time'],
                      y=node['labels'], mask=1, type=0)
    else:
        tree.add_node(idx2idx_dict[index], u=node['features'][0], x=node['features'][1], time=node['time'],
                      y=node['labels'], mask=1, type=5)

    if index > 0 and flag_dict[index] > 0:
        flag_dict[index] -= 1  # already play as parent node
        for i in range(nary, 0, -1):
            if index - i < 0:  # fictitious node
                node_id = tree.number_of_nodes()
                tree.add_node(node_id, u=node['features'][0], x=trajectory[-1]['features'][1], time=node['time'],
                              y=[-1] * 3, mask=0, type=-1)
                tree.add_edge(node_id, idx2idx_dict[index])  # src -> dst
                max_index = len(trajectory) - 1
            else:
                child_idx = add_children_out(tree, trajectory, index - i, idx2idx_dict, flag_dict, nary)
                max_index = child_idx if max_index < child_idx else max_index
                tree.add_edge(idx2idx_dict[index - i], idx2idx_dict[index])  # src -> dst

        node_id = tree.number_of_nodes()
        tree.add_node(node_id, u=node['features'][0], x=node['features'][1], time=node['time'],
                      y=node['labels'], mask=1, type=3)
        tree.add_edge(node_id, idx2idx_dict[index])  # src -> dst
        for i in range(1, 3):
            node_id = tree.number_of_nodes()
            tree.add_node(node_id, u=node['features'][0], x=node['features'][i + 1], time=node['time'],
                          y=node['labels'], mask=1, type=i)
            tree.add_edge(node_id, idx2idx_dict[index])  # src -> dst

    # change node label
    tree.add_node(idx2idx_dict[index], u=node['features'][0], x=node['features'][1], time=node['time'],
                  y=trajectory[max_index]['labels'], mask=1, type=0)
    return max_index


def add_true_node(tree, trajectory, index, parent_node_id, nary):
    for i in range(nary - 1, 0, -1):
        if index - i >= 0:
            node_id = tree.number_of_nodes()
            node = trajectory[index - i]
            tree.add_node(node_id, x=node['features'], time=node['time'], y=node['labels'], mask=1, mask2=0, type=1)
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
    tree.add_node(node_id, x=[0] * 4, time=0, y=period_label, mask=0, mask2=1, type=-1)

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


if __name__ == "__main__":
    # dataset = 'NYC'
    # train_df = pd.read_csv(f'dataset/{dataset}/{dataset}_train.csv')
    #
    # # User id to index
    # uid_list = [str(uid) for uid in list(set(train_df['user_id'].to_list()))]
    # user_id2idx_dict = dict(zip(uid_list, range(len(uid_list))))
    # # POI id to index
    # POI_list = list(set(train_df['POI_id'].tolist()))
    # POI_list.sort()
    # POI_id2idx_dict = dict(zip(POI_list, range(len(POI_list))))
    # # Cat id to index
    # cat_list = list(set(train_df['POI_catid'].tolist()))
    # cat_list.sort()
    # cat_id2idx_dict = dict(zip(cat_list, range(len(cat_list))))
    #
    # data_train = np.column_stack((train_df['longitude'], train_df['latitude']))
    # kmeans_train = KMeans(n_clusters=50)
    # kmeans_train.fit(data_train)
    # train_df['coo_label'] = kmeans_train.labels_
    #
    # # Build dataset
    # map_set = (user_id2idx_dict, POI_id2idx_dict, cat_id2idx_dict)
    # train_dataset = TrajectoryTrainDataset(train_df, map_set)
    # train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False,
    #                               pin_memory=True, num_workers=0, collate_fn=lambda x: x)
    #
    # for b_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training"):
    #     in_tree_batcher, out_tree_batcher = [], []
    #     for trajectory, label in batch:
    #         construct_MobilityTree(trajectory, label, 5, True, 'in')
    #     break
    f = [0] * 4
    trajectory = [[[{'features': f, 'time': 0, 'labels': [1]}, {'features': f, 'time': 0, 'labels': [2]},
                    {'features': f, 'time': 0, 'labels': [3]}, {'features': f, 'time': 0, 'labels': [4]},
                    {'features': f, 'time': 0, 'labels': [5]}, {'features': f, 'time': 0, 'labels': [6]}],
                   [],
                   [],
                   [{'features': f, 'time': 0, 'labels': [4, 1]}]],
                  [[],
                   [{'features': f, 'time': 0, 'labels': [1]}, {'features': f, 'time': 0, 'labels': [2]},
                    {'features': f, 'time': 0, 'labels': [3]}, {'features': f, 'time': 0, 'labels': [4]},
                    {'features': f, 'time': 0, 'labels': [5]}, {'features': f, 'time': 0, 'labels': [6]}],
                   [{'features': f, 'time': 0, 'labels': [1]}, {'features': f, 'time': 0, 'labels': [2]},
                    {'features': f, 'time': 0, 'labels': [3]}, {'features': f, 'time': 0, 'labels': [4]},
                    {'features': f, 'time': 0, 'labels': [5]}, {'features': f, 'time': 0, 'labels': [6]}],
                   []],
                  [[],
                   [],
                   [],
                   []],
                  [[],
                   [],
                   [],
                   []],
                  [[],
                   [],
                   [],
                   []],
                  [[],
                   [],
                   [],
                   []]]

    label = [[[1] * 3], [[2] * 3], [[3] * 3], [[4] * 3], [[5] * 3], [[6] * 3]]
    construct_MobilityTree(trajectory, label, 5, True, 'in')
