import torch
from torch import nn

x = torch.ones(3)
torch.save(x, 'x.pt')

x2 = torch.load('x.pt')
print(x2)


# 读写模型
# state_dict

class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden)
        return self.output(a)


net = mlp()
print(net.state_dict())
