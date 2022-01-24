from models import get_cora_model
from dig.xgraph.method import SubgraphX
from dig.xgraph.method.subgraphx import find_closest_node_result
import torch
from custom_subgraphx import subgraphx


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 55 long
# 20 short
node_idx = 20
model, dataset = get_cora_model()
data = dataset[0]
logits = model(data.x, data.edge_index)
prediction = logits[node_idx].argmax(-1).item()
result = subgraphx(data.x, data.edge_index, model, node_idx=node_idx)
print(result)
# # Explain with subgraphx library
# explainer = SubgraphX(model, num_classes=4, device=device, explain_graph=False, reward_method='nc_mc_l_shapley')
# _, explanation_results, related_preds = explainer(data.x, data.edge_index, node_idx=node_idx, max_nodes=5)
# result = find_closest_node_result(explanation_results[prediction], max_nodes=5)
# print(result)