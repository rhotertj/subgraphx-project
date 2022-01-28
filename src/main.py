from sqlalchemy import false
from models import get_cora_model
from dig.xgraph.method import SubgraphX, PGExplainer
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
prediction1 = logits[node_idx]
print("Prediction:", prediction1)

fids = []
for _ in range(5):
    result = subgraphx(data.x, data.edge_index, model, node_idx=node_idx, M=40, Nmin=8, L=2)
    print(result)

    node_features = data.x.clone()
    for n in result:
        node_features[n] = torch.zeros_like(data.x[n])
    logits = model(node_features, data.edge_index)
    prediction2 = logits[node_idx]
    print("Prediction", prediction2)

    #c = prediction1.argmax(-1)
    fidelity = prediction1 - prediction2
    print("Fidelity", fidelity.item())
    fids.append(fidelity)

# print("Fidelity mean:", sum(fids) / len(fids))
print(fids)
print(torch.stack(fids).mean(dim=0))
# # Explain with subgraphx library
explainer = PGExplainer(model, dataset.num_node_features, device, False)
result = explainer(data.x, data.edge_index)
explainer = SubgraphX(model, num_classes=dataset.num_classes, device=device, explain_graph=False, reward_method='mc_shapley')
_, explanation_results, related_preds = explainer(data.x, data.edge_index, node_idx=int(node_idx), max_nodes=8)
# result = find_closest_node_result(explanation_results[prediction], max_nodes=8)
print(explanation_results)