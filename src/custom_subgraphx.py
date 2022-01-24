import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph
import itertools
from tqdm import tqdm
import random

class Node:

    def __init__(self, graph, parent) -> None:
        self.subgraph = graph
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
        # print("nodes to prune:", available_nodes_idx)
        successors = []
        for node_to_prune in available_nodes_idx:
            if node_to_prune in self.children:
                successors.append(self.children[node_to_prune])
            else:
                new_sub = self.subgraph.copy()
                new_sub[node_to_prune] = np.zeros_like(self.subgraph[node_to_prune])
                # if node already exists by different action combinations, use this node
                # s.t. pruning 1 and 3 results in same node as pruning 3 and 1
                for k, v in self.children.items():
                    if (v.subgraph == new_sub).all():
                        self.children[node_to_prune] = v
                successor = Node(new_sub, parent=self)
                self.children[node_to_prune] = successor
                successors.append(successor)
        return successors


    def upper_bound(self, alternative_action_samples, l=5):
        return l * self.score * (np.sqrt(alternative_action_samples) / (1 + self.n_samples))

# main subgraphx algortihm
def subgraphx(graph, edge_index, model, M=20, Nmin=15, node_idx=None, L=2):
    # graph classification
    if node_idx is None:
        root = Node(graph, parent=None)
    # node classification
    if isinstance(node_idx, int):
        subgraph = np.zeros_like(graph)
        neighbors, *_ = k_hop_subgraph(node_idx, L, edge_index)
        for i in neighbors.tolist():
            subgraph[i] = graph[i]
        root = Node(subgraph, None)
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
                score = compute_score(edge_index, child.subgraph, child.get_node_idx(), model)
                child.score = score
            # mcts selection of next pruning action
            sum_samples = sum([child.n_samples for child in children])
            selection_critera = [child.mean + child.upper_bound(sum_samples) for child in children]
            next_node_idx = np.argmax(selection_critera)
            current_node = children[next_node_idx]
            current_node.total_reward += score
            current_node.n_samples += 1

        leaves.append(current_node)
    # return subgraph with highest expected shapley contribution
    print(root)

    best_node_idx = np.argmax([l.mean for l in leaves])
    return np.unique(np.where(leaves[best_node_idx].subgraph)[0])


# algorithm to rate subgraph, reward with shapley:
def compute_score(edge_index, subgraph, subgraph_idx, model, L=1, T=10):
    subgraph_idx = torch.tensor(subgraph_idx)
    neighbors, *_ = k_hop_subgraph(subgraph_idx, L, edge_index)

    players = subgraph.copy()
    for i in neighbors:
        players[i] = subgraph[i]
    players = torch.tensor(players)
    pred_player = model(players, edge_index)
    shaps = []
    for i in range(T):
        # sample coalition from neighbors
        coalition_idx = sample_coalition(neighbors.tolist())
        coalition = np.zeros_like(subgraph)
        for i in coalition_idx:
            coalition[i] = subgraph[i]
        coalition = torch.tensor(coalition)
        pred_coalition = model(coalition, edge_index)
        shap = pred_player - pred_coalition
        shaps.append(shap.detach().numpy())
    
    return np.mean(shaps)



def powerset(iterable):
    # too memory hungry
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return [set(c) for c in itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))]

def sample_coalition(coalition_idx):
    coalition_len = random.choice(list(range(1, len(coalition_idx))))
    coalition = random.choices(coalition_idx, k=coalition_len)
    return coalition