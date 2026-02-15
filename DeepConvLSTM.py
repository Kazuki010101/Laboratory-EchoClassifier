import torch
import torch.nn as nn
import torch.nn.functional as F

train_on_gpu = torch.cuda.is_available()
train_on_gpu = False


class DeepConvLSTM(nn.Module):
    
    def __init__(self, n_hidden=128, n_layers=1, n_filters=64, 
                 n_classes=18, filter_size=5, window_size = 500, channels=3, drop_prob=0.5, batch_size=64):
        super(DeepConvLSTM, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.channnels = channels
        self.window_size = window_size
        self.batch_size = batch_size
             
        self.conv1 = nn.Conv1d(channels, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)
        
        self.lstm1  = nn.LSTM(n_filters, n_hidden, n_layers, batch_first=True)
        self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True)
        
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):
        n_window_size = 1
        batch_size = x.size(0)
        
        hidden = self.init_hidden(batch_size)
        
        try:
            x = x.view(batch_size, self.channnels, self.window_size)
        except RuntimeError:
            x = x.view(self.batch_size, self.channnels, -1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.permute(0, 2, 1)  
        x, hidden = self.lstm1(x, hidden)
        x, hidden = self.lstm2(x, hidden)
        
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])
        
        return x
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        weight = next(self.parameters()).data
        
        if train_on_gpu:
            hidden = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda(),
                      weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda())
        else:
            hidden = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
                      weight.new_zeros(self.n_layers, batch_size, self.n_hidden))
        
        return hidden
