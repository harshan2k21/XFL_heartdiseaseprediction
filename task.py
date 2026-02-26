import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input: 13 features from Heart Disease dataset
        self.fc1 = nn.Linear(13, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2) # Outcome: Disease or No Disease

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x) # We'll use CrossEntropyLoss which includes Softmax

# train and test functions remain mostly the same, 
# but we'll use CrossEntropyLoss instead of NLL_Loss.