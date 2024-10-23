import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfModelingMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10, aw=10):
        super(SelfModelingMLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
        self.aw = aw  # Auxiliary Weight
        self.self_model = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        hidden_activations = F.relu(self.hidden(x))
        logits = self.output(hidden_activations)
        self_pred = self.self_model(hidden_activations)
        return logits, self_pred, hidden_activations

    def get_mean_l2_norm(self):
        mean_l2_norm = 0
        for param in self.parameters():
            mean_l2_norm += torch.linalg.norm(param, 2)
        return mean_l2_norm / len(list(self.parameters()))
