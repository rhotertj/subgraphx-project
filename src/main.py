from models import get_cora_model
from dig.xgraph.method import SubgraphX
from dig.xgraph.method.subgraphx import find_closest_node_result
import torch
from custom_subgraphx import subgraphx


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 55 long
# 20 short
node_idx = 11
model, dataset = get_cora_model("karate")
data = dataset[0]
logits = model(data.x, data.edge_index)
prediction = logits[node_idx]
print("Prediction:", prediction)
result = subgraphx(data.x, data.edge_index, model, node_idx=node_idx, M=40, Nmin=8, L=2)
print(result)
node_features = torch.zeros_like(data.x)
for n in result:
    node_features[n] = data.x[n]
logits = model(node_features, data.edge_index)
prediction = logits[node_idx]
print("Prediction", prediction)
# # Explain with subgraphx library
# explainer = SubgraphX(model, num_classes=dataset.num_classes, device=device, explain_graph=False)
# _, explanation_results, related_preds = explainer(data.x, data.edge_index, node_idx=node_idx, max_nodes=8)
# result = find_closest_node_result(explanation_results[prediction], max_nodes=8)
# print(result)