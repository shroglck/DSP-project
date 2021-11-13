import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self):
        super(ANN,self).__init__()
        self.flatten=nn.Flatten()
        self.dense_layers= nn.Sequential(
            nn.Linear(313*128,256),
            
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(128,3),
            )
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        x=self.flatten(x)
        x=self.dense_layers(x)
        prediction=self.softmax(x)
        return prediction
    