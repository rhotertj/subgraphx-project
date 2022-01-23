
# TODO: Shapley computation for different subgraph coalition
# TODO: MCTS for pruning subgraphs
# TODO: otline algorithm
import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph
import itertools
import random

class Node:

    def __init__(self, graph, parent) -> None:
        self.subgraph = graph
        self.total_reward = 0
        self.n_samples  = 0
        # last score
        self.score = 0
        self.parent = parent

    @property
    def mean(self):
        if self.n_samples == 0:
            return 0
        else:
            return self.total_reward / self.n_samples

    def nodes_left(self):
        # Return number of possible nodes to prune
        return len(np.where(sum(self.subgraph, axis=1))[0])

    def possible_successors(self):
        # compute possible subgraphs by prubning one node and return nodes
        # set parent
        available_nodes_idx = np.where(sum(self.subgraph, axis=1))[0]
        successors = []
        for i in available_nodes_idx:
            new_sub = self.subgraph.copy()
            new_sub[i] = np.zeros_like(self.subgraph[i])
            successors.append(Node(new_sub, parent=self))
        return successors


    def upper_bound(self, alternative_action_samples, l=5):
        return l * self.score * (np.sqrt(alternative_action_samples) / (1 + self.n_samples))

# main subgraphx algortihm
def subgraphx(graph, edge_index, model, M=20, Nmin=5):
    leaves = []
    root = Node(graph, parent=None)
    for i in range(M):
        current_node = root
        while current_node.nodes_left() > Nmin:
            children = current_node.possible_successors()
            for child in children:
                score = score(edge_index, child.subgraph, model)
                child.score = score
            sum_samples = sum([child.n_samples for child in children])
            selection_critera = [child.mean + child.upper_bound(sum_samples) for child in children]
            next_node_idx = np.argmax(selection_critera)
            current_node = children[next_node_idx]
            current_node.total_reward += score
            current_node.n_samples += 1

        leaves.append(current_node)
    best_node_idx = np.argmax([l.mean for l in leaves])
    return leaves[best_node_idx].subgraph


# algorithm to rate subgraph, reward with shapley:
def score(edge_index, subgraph, model, L=3, T=1000):
    neighbors = []
    node_idx = None # node idx of subgraph nodes in graph
    neighbors, _, mapping, _ = k_hop_subgraph(node_idx, L, edge_index)
    print(mapping)
    # add neighbors to subgraph nodes
    # TODO: correct mapping
    all_node_indices = list(range(subgraph.shape[0]))
    all_coalitions = powerset(all_node_indices)
    players = subgraph.clone()
    for i,j in enumerate(mapping):
        players[j] = neighbors[i]
    pred_player = model(players, edge_index)
    shaps = []
    for i in range(T):
        # sample coalition from neighbors
        coalition_idx = random.choice(all_coalitions)
        coalition = subgraph[np.array(coalition_idx)]
        pred_coalition = model(coalition, edge_index)
        shap = pred_player - pred_coalition
        shaps.append(shap)
    
    return np.mean(shaps)



def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return [set(c) for c in itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))]