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
    
    def get_node_idx(self):
        return np.where(np.sum(self.subgraph, axis=1))[0]

    def nodes_left(self):
        # Return number of possible nodes to prune
        return len(np.where(np.sum(self.subgraph, axis=1))[0])

    def possible_successors(self):
        # compute possible subgraphs by prubning one node and return nodes
        # set parent
        available_nodes_idx = np.where(np.sum(self.subgraph, axis=1))[0]
        print("nodes to prune:", available_nodes_idx)
        successors = []
        for i in available_nodes_idx:
            new_sub = self.subgraph.copy()
            new_sub[i] = np.zeros_like(self.subgraph[i])
            successors.append(Node(new_sub, parent=self))
        return successors


    def upper_bound(self, alternative_action_samples, l=5):
        return l * self.score * (np.sqrt(alternative_action_samples) / (1 + self.n_samples))

# main subgraphx algortihm
def subgraphx(graph, edge_index, model, M=20, Nmin=5, node_idx=None):
    if isinstance(node_idx, int):
        subgraph = np.zeros_like(graph)
        neighbors, *_ = k_hop_subgraph(node_idx, 1, edge_index)
        for i in neighbors.tolist():
            subgraph[i] = graph[i]
        root = Node(subgraph, None)

    elif len(node_idx) == 2:
        # TODO Set subgraph of k-hop neighborhood from both nodes as root
        exit()
    else:
        root = Node(graph, parent=None)

    leaves = []
    for i in range(M):
        #TODO: Save tree, only compute new children if not visited yet
        print("MCTS", i)
        current_node = root
        while current_node.nodes_left() > Nmin:
            cn_left = current_node.nodes_left() 
            children = current_node.possible_successors()
            for child in children:
                print(cn_left, child.nodes_left())
                score = compute_score(edge_index, child.subgraph, child.get_node_idx(), model)
                child.score = score
            sum_samples = sum([child.n_samples for child in children])
            selection_critera = [child.mean + child.upper_bound(sum_samples) for child in children]
            next_node_idx = np.argmax(selection_critera)
            current_node = children[next_node_idx]
            current_node.total_reward += score
            current_node.n_samples += 1

        leaves.append(current_node)
    best_node_idx = np.argmax([l.mean for l in leaves])
    return np.where(leaves[best_node_idx].subgraph)


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