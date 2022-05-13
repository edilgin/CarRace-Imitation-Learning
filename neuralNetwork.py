import torch.nn as nn

class myNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=12, kernel_size=(3,3), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=12,out_channels=24, kernel_size=(3,3), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=24,out_channels=48, kernel_size=(3,3), stride=(2,2)),
            nn.Flatten()
        )


        # might need to set bias to false


        self.linear = nn.Sequential(
            nn.Linear(in_features=5808, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x
