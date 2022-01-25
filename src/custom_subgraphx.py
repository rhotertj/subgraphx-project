import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph
import itertools
from tqdm import tqdm
import random
import time

from sgx_utils import (
    khop_node_edges,
    khop_node_edges,
    largest_connected_subgraph,
    sample_coalition,
    subgraph_by_node_removal
)

class Node:

    def __init__(self, graph, edges, parent) -> None:
        self.subgraph = graph
        self.edges = edges
        self.total_reward = 0
        self.n_samples  = 0
        # last score
        self.score = 0
        self.parent = parent
        self.children = {}

    @property
    def mean(self):
        if self.n_samples == 0:
            return 0
        else:
            return self.total_reward / self.n_samples
    
    def __repr__(self) -> str:
        r = ""
        r += f"Node (reward={self.mean}, sampled={self.n_samples}) with {self.nodes_left()} graph nodes and {len(self.children)} children:"
        # for action, node in self.children.items():
        #     r += f"{action} -> {node.__repr__()}\n"
        return r

    def get_node_idx(self):
        return np.where(np.sum(self.subgraph, axis=1))[0]

    def nodes_left(self):
        # Return number of possible nodes to prune
        return len(np.where(np.sum(self.subgraph, axis=1))[0])

    def possible_successors(self):
        # compute possible subgraphs by prubning one graph node and return mcts nodes
        # set parent
        available_nodes_idx = np.where(np.sum(self.subgraph, axis=1))[0]
        
        successors = []
        for node_to_prune in available_nodes_idx:
            # print("prune node:", node_to_prune, "graph sum", sum(available_nodes_idx))
            if node_to_prune in self.children:
                successors.append(self.children[node_to_prune])
            else:
                # print("not there")
                print("NODE TO PRUNE", node_to_prune, available_nodes_idx)
                new_sub, new_edges = subgraph_by_node_removal(self.subgraph, self.edges, node_to_prune)
                # Check if subgraph still connected
                new_sub, new_edges = largest_connected_subgraph(new_sub, new_edges)
                # if node already exists by different action combinations, use this node
                # s.t. pruning 1 and 3 results in same node as pruning 3 and 1
                successor = None
                for k, v in self.children.items():
                    if (v.subgraph == new_sub).all():
                        print("no new node")
                        successor = v
                if successor is None:
                    successor = Node(new_sub, new_edges, parent=self)
                self.children[node_to_prune] = successor
                successors.append(successor)
        return successors


    def upper_bound(self, alternative_action_samples, l=5):
        return l * self.score * (np.sqrt(alternative_action_samples) / (1 + self.n_samples))

global_time = 0
def check_time(location):
    global_time = globals()["global_time"]
    now = time.time()
    print(f"{location} : {now - global_time}")
    globals()["global_time"] = now

# main subgraphx algortihm
def subgraphx(graph, edge_index, model, M=20, Nmin=4, node_idx=None, L=1):
    # graph classification
    if node_idx is None:
        root = Node(graph, edge_index, parent=None)
    # node classification
    if isinstance(node_idx, int):
        nodes, edges = khop_node_edges(node_idx, graph, edge_index, L)
        root = Node(nodes, edges, None)
    # link prediction
    elif len(node_idx) == 2:
        # TODO Set subgraph of k-hop neighborhood from both nodes as root
        exit()
    else:
        raise Exception("Invalid parameters")

    leaves = []
    for i in tqdm(range(M)):
        current_node = root
        # Still nodes to prune
        while current_node.nodes_left() > Nmin:
            children = current_node.possible_successors()
            for child in children:
                # shapley contribution of pruned subgraph wrt full subgraph
                score = compute_score(child.edges, child.subgraph, child.get_node_idx(), model, node_idx=node_idx)
                child.score = score
                print(score)
            # mcts selection of next pruning action
            sum_samples = sum([child.n_samples for child in children])
            selection_critera = [child.mean + child.upper_bound(sum_samples) for child in children]
            print(selection_critera)
            next_node_idx = np.argmax(selection_critera)
            current_node = children[next_node_idx]
        iter_node = current_node
        while True:
            iter_node.total_reward += score
            iter_node.n_samples += 1
            if iter_node.parent is None:
                break
            iter_node = iter_node.parent

        leaves.append(current_node)
    # return subgraph with highest expected shapley contribution
    [print(c) for c in root.children.values()]
    best_node_idx = np.argmax([l.mean for l in leaves])
    return np.unique(np.where(leaves[best_node_idx].subgraph)[0])


# algorithm to rate subgraph, reward with shapley:
def compute_score(edge_index, subgraph, subgraph_idx, model, L=1, T=100, node_idx=None):
    subgraph_idx = torch.tensor(subgraph_idx)
    try:
        neighbors, *_ = k_hop_subgraph(subgraph_idx, L, edge_index)
    except Exception as e:
        print(e)
        print(subgraph_idx, subgraph_idx.shape)
        print(edge_index, edge_index.shape)
        print(node_idx)
    shaps = []
    for i in range(T):
        # sample coalition from neighbors
        coalition_idx = sample_coalition(neighbors.tolist())
        # set features to zero except for coalition and subgraph
        sg_and_coal = subgraph.copy()
        for i in coalition_idx:
            sg_and_coal[i] = subgraph[i]
        sg_and_coal = torch.tensor(sg_and_coal)
        pred_player = torch.max(model(sg_and_coal, edge_index), dim=1)[0]
        if not node_idx is None:
            pred_player = pred_player[node_idx]
        # set features to zero except for coalition
        coalition = np.zeros_like(subgraph)
        for i in coalition_idx:
            coalition[i] = subgraph[i]
        coalition = torch.tensor(coalition)
        pred_coalition = torch.max(model(coalition, edge_index), dim=1)[0]
        if not node_idx is None:
            pred_coalition = pred_coalition[node_idx]
        shap = pred_player - pred_coalition
        shaps.append(shap.detach().numpy())
    
    return np.mean(shaps)
