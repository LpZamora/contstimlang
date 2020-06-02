import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size):
        super(RNNModel, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # Number of hidden dimensions
        self.hidden_size = hidden_size
        
        # Number of hidden layers
        self.num_layers = num_layers
        
        # RNN
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, h0):
        
        x = self.embed(x).requires_grad_()  

        # One time step
        out, hn = self.rnn(x, h0)     
        
        out = self.fc(out) 
        
        #print(out.shape)
        return out, hn