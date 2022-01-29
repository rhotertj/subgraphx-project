from sqlalchemy import false
from models import get_cora_model
from dig.xgraph.method import SubgraphX, PGExplainer
from dig.xgraph.method.subgraphx import find_closest_node_result
from torch_geometric.nn import GNNExplainer
import torch
import numpy as np
from custom_subgraphx import subgraphx
from sgx_utils import subgraph_by_node_removal

# Fidelity: Remove important features and expect large change in prediction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 55 long
# 20 short
node_idx = 11
model, dataset = get_cora_model("karate")
data = dataset[0]
logits = model(data.x, data.edge_index)
prediction1 = torch.exp(logits[node_idx])
print("Prediction:", prediction1)

fids = []
for _ in range(5):
    result = subgraphx(data.x, data.edge_index, model, node_idx=node_idx, M=40, Nmin=8, L=2)
    nodes, edges = data.x.clone().numpy(), data.edge_index.clone()
    for node_to_remove in result:
        nodes, edges = subgraph_by_node_removal(nodes, edges, node_to_remove)
    logits = model(torch.tensor(nodes), edges)
    prediction2 = torch.exp(logits[node_idx])
    print("Prediction", prediction2)

    #c = prediction1.argmax(-1)
    fidelity = prediction1 - prediction2
    print("Fidelity", fidelity)
    fids.append(fidelity)

# print("Fidelity mean:", sum(fids) / len(fids))
print(fids)
print("Fidelity subgraphx =" ,torch.stack(fids).mean(dim=0))

### explain with pyg
explainer = GNNExplainer(model, epochs=200, return_type='log_prob')

gnn_fids = []
for _ in range(5):
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index)
    print(data.edge_index.shape)
    masked_x = data.x.clone()
    for i, f in enumerate(node_feat_mask):
        if f > 0.5: # above 0.5 important
            masked_x[:, i] = torch.zeros_like(data.x[:, i])

    masked_edges = []
    edge_mask = edge_mask.reshape(1, -1).T

    for i, e in enumerate(edge_mask):
        if e.item() < 0.5: # above 0.5 important
            masked_edges.append(data.edge_index.T[i])

    masked_edges = torch.stack(masked_edges).T
    print(masked_edges.shape)
    logits = model(masked_x, masked_edges)
    prediction3 = torch.exp(logits[node_idx])
    fidelity = prediction1 - prediction3
    print("Fidelity", fidelity)
    gnn_fids.append(fidelity)

print(gnn_fids)
print("Fidelity gnn =", torch.stack(gnn_fids).mean(dim=0))


# # Explain with subgraphx library
# explainer = PGExplainer(model, dataset.num_node_features, device, False)
# result = explainer(data.x, data.edge_index)
# explainer = SubgraphX(model, num_classes=dataset.num_classes, device=device, explain_graph=False, reward_method='mc_shapley')
# _, explanation_results, related_preds = explainer(data.x, data.edge_index, node_idx=int(node_idx), max_nodes=8)
# # result = find_closest_node_result(explanation_results[prediction], max_nodes=8)
# print(explanation_results)