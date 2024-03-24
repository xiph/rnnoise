import torch
from torch import nn
import torch.nn.functional as F

class RNNoise(nn.Module):
    def __init__(self, input_dim=45, output_dim=22, cond_size=128, gru_size=256):
        super(RNNoise, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cond_size = cond_size
        self.gru_size = gru_size
        self.conv1 = nn.Conv1d(input_dim, cond_size, kernel_size=3, padding='valid')
        self.conv2 = nn.Conv1d(cond_size, gru_size, kernel_size=3, padding='valid')
        self.gru1 = nn.GRU(self.gru_size, self.gru_size, batch_first=True)
        self.gru2 = nn.GRU(self.gru_size, self.gru_size, batch_first=True)
        self.gru3 = nn.GRU(self.gru_size, self.gru_size, batch_first=True)
        self.dense_out = nn.Linear(self.gru_size, self.output_dim)
        nb_params = sum(p.numel() for p in self.parameters())
        print(f"model: {nb_params} weights")

    def forward(self, features, states=None):
        #print(states)
        device = features.device
        batch_size = features.size(0)
        if states is None:
            gru1_state = torch.zeros((1, batch_size, self.gru_size), device=device)
            gru2_state = torch.zeros((1, batch_size, self.gru_size), device=device)
            gru3_state = torch.zeros((1, batch_size, self.gru_size), device=device)
        else:
            gru1_state = states[0]
            gru2_state = states[1]
            gru3_state = states[2]
        tmp = features.permute(0, 2, 1)
        tmp = torch.tanh(self.conv1(tmp))
        tmp = torch.tanh(self.conv2(tmp))
        tmp = tmp.permute(0, 2, 1)

        gru1_out, gru1_state = self.gru1(tmp, gru1_state)
        gru2_out, gru2_state = self.gru2(gru1_out, gru2_state)
        gru3_out, gru3_state = self.gru3(gru2_out, gru3_state)
        return torch.sigmoid(self.dense_out(gru3_out)), [gru1_state, gru2_state, gru3_state]
