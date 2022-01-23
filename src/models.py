# Taken from Exercise 9

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from tqdm import tqdm



class GCN(torch.nn.Module):
  
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=False)

    def forward(self, x, edge_index, ):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def train_model(self, data, x, edge_index, train_mask, optimizer, epochs):
        print("Training the model...")
        for epoch in tqdm(range(1, epochs)):
            self.train()
            optimizer.zero_grad()
            log_logits = self(x, edge_index)
            loss = F.nll_loss(log_logits[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()


def get_cora_model():

    dataset = "Cora"
    transform = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
    dataset = Planetoid("./", dataset, transform=transform)

    data = dataset[0]

    in_channels = dataset.num_features
    hidden_channels = 16
    out_channels = dataset.num_classes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(in_channels, hidden_channels, out_channels).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
    model.train_model(data, x, edge_index, data.train_mask, optimizer, 200)

    return model, dataset



