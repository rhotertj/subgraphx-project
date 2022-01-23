from models import get_cora_model
from dig.xgraph.method import SubgraphX
from dig.xgraph.method.subgraphx import find_closest_node_result
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
node_idx = 3
model, dataset = get_cora_model()
data = dataset[0]
logits = model(data.x, data.edge_index, data.edge_weight)
prediction = logits[node_idx].argmax(-1).item()

# Explain with subgraphx library
explainer = SubgraphX(model, num_classes=4, device=device, explain_graph=False, reward_method='nc_mc_l_shapley')
_, explanation_results, related_preds = explainer(data.x, data.edge_index, data.edge_weight, node_idx=node_idx, max_nodes=5)
result = find_closest_node_result(explanation_results[prediction], max_nodes=5)
print(result)