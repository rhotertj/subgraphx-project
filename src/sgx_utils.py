import torch
import numpy as np
import random
from torch_geometric.utils import k_hop_subgraph



def largest_connected_subgraph(nodes, edges):
    all_nodes_subgraph = np.unique(edges)
    first_node_subgraph, *_ = k_hop_subgraph(int(all_nodes_subgraph[0]), len(all_nodes_subgraph), edges)
    new_nodes = nodes
    new_edges = edges
    if not len(first_node_subgraph) == len(all_nodes_subgraph):
        print("Not connected anymore")
        # look for largest connected subgraph
        subgraph_sizes = []
        for query_node in all_nodes_subgraph:
            query_subgraph, *_ =  k_hop_subgraph(int(query_node), len(all_nodes_subgraph), edges)
            subgraph_sizes.append((query_node, len(query_subgraph)))
        largest_sg_node = sorted(subgraph_sizes, reverse=True, key=lambda nl: nl[1])[0]
        
        new_nodes, new_edges = khop_node_edges(largest_sg_node, nodes, edges, len(all_nodes_subgraph))
        for n in np.where(np.sum(new_nodes, axis=1))[0]:
                    assert n in new_edges, f"{np.where(np.sum(nodes, axis=1))[0], edges, all_nodes_subgraph}"
        
    return new_nodes, new_edges


def subgraph_by_node_removal(old_nodes, old_edges, node_to_remove):
    new_nodes = old_nodes.copy()
    new_nodes[node_to_remove] = np.zeros_like(old_nodes[node_to_remove])
    # one column is one edge
    # remove edges containing only pruned node
    edges_to_keep_bool = np.all(np.array(old_edges != node_to_remove), axis=0)
    edges_to_keep_idx = np.where(edges_to_keep_bool)[0]
    new_edges = old_edges.clone()[:, edges_to_keep_idx]
    all_valid_nodes_idx = np.unique(new_edges)
    newer_nodes = np.zeros_like(new_nodes)
    # filter nodes that were only connected to the pruned node
    for i in all_valid_nodes_idx:
        newer_nodes[i] = new_nodes[i]
    for n in np.where(np.sum(newer_nodes, axis=1))[0]:
        assert n in np.unique(new_edges), f"{n, new_edges, node_to_remove}"
    assert not node_to_remove in np.unique(new_edges)
    for n in np.unique(new_edges):
        assert n in np.where(np.sum(newer_nodes, axis=1))[0]
    return newer_nodes, new_edges

def khop_node_edges(node_idx, old_nodes, old_edges, k):
    new_nodes = np.zeros_like(old_nodes)
    neighbors, *_ = k_hop_subgraph(node_idx, k, old_edges)
    new_edges = []
    for i in neighbors.tolist():
        new_nodes[i] = old_nodes[i]
        # allow only edges within subgraph
    for i, edge in enumerate(old_edges.T):
        u, v = edge
        if u in neighbors and v in neighbors:
            new_edges.append(old_edges.T[i])
    new_edges = torch.stack(new_edges).T
    
    return new_nodes, new_edges

def sample_coalition(coalition_idx):
    coalition_len = random.choice(list(range(1, len(coalition_idx))))
    coalition = random.choices(coalition_idx, k=coalition_len)
    return coalition