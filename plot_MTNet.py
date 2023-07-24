import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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
    node_labels = nx.get_node_attributes(g, 'time')
    nx.draw_networkx_labels(g, pos, labels=node_labels, font_color='green')
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


def construct_MobilityTree(trajectory, nary, need_plot, tree_type):
    tree = nx.DiGraph()
    idx2idx_dict = {}
    flag_dict = dict(zip(range(len(trajectory)), np.full(len(trajectory), 2)))

    start_index = len(trajectory) - 1
    if tree_type == 'in':  # in tree
        add_children(tree, trajectory, start_index, idx2idx_dict, flag_dict, nary)
    elif tree_type == 'out':  # out tree
        add_children_out(tree, trajectory, start_index, idx2idx_dict, flag_dict, nary)
    else:
        print("Tree type wrong!")

    if need_plot:
        plot_tree(tree)  # optional


if __name__ == "__main__":
    trajectory = []
    for i in range(8):
        checkin = {'features': [f'{i}_0', f'{i}_1', f'{i}_2', f'{i}_3'], 'time': f'{i}', 'labels': f'  {i}'}
        trajectory.append(checkin)

    # construct_MobilityTree(trajectory, 3, True, 'in')
    # construct_MobilityTree(trajectory, 3, True, 'out')
    construct_MobilityTree(trajectory, 1, True, 'in')
    construct_MobilityTree(trajectory, 1, True, 'out')
    # construct_MobilityTree(trajectory, 2, True, 'in')
    # construct_MobilityTree(trajectory, 2, True, 'out')
