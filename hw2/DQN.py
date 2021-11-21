import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, field_size):
        super(DQN, self).__init__()
        self.field_size = field_size
        
        self.conv = nn.Conv2d(1, self.field_size * self.field_size, kernel_size = self.field_size)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class DQN_noughts(nn.Module):
    def __init__(self, field_size, n_chanels):
        super(DQN_noughts, self).__init__()
        self.field_size = field_size
        self.n_chanels = n_chanels
        
        self.conv = nn.Conv2d(1, self.n_chanels, kernel_size = self.field_size)
        self.l = nn.Linear(self.n_chanels, self.field_size * self.field_size)
    def forward(self, x):
        x = F.relu(self.conv(x))
        b = x.size()[0]
        x = x.view(b, -1)
        x = self.l(x)
        return x