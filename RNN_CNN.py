import torch
import torch.nn as nn

class LSTM_CNN(nn.Module):
    def __init__(self,input_size,timesteps):
        super(LSTM_CNN,self).__init__()
        self.input_size=input_size
        self.timesteps=timesteps
        self.lstm1=nn.LSTM(input_size=self.input_size,hidden_size=64,num_layers=1,batch_first=True)
        self.cnn1=nn.Conv1d(in_channels=self.timesteps,out_channels=self.timesteps,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()
        self.lstm2=nn.LSTM(input_size=64,hidden_size=3,num_layers=1,batch_first=True)
        self.softmax=nn.Softmax(dim=2)
    def forward(self,input):
        x,(h,c)=self.lstm1(input)
        x=self.cnn1(x)
        x=self.relu(x)
        x,_=self.lstm2(x)
        x=self.softmax(x)
        return x



        
        