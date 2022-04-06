import torch.nn as nn

# RNN based language model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h, epoch):

        # Embed word ids to vectors
        x = self.embed(x)

        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))

        # Decode hidden states of all time steps
        out = self.linear(out)

        return out, (h, c)


class RNNModel(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size):
        super(RNNModel, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)

        # Number of hidden dimensions
        self.hidden_size = hidden_size

        # Number of hidden layers
        self.num_layers = num_layers

        # RNN
        self.rnn = nn.RNN(
            embed_size, hidden_size, num_layers, batch_first=True, nonlinearity="relu"
        )

        # Readout layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h0):

        x = self.embed(x).requires_grad_()

        # One time step
        out, hn = self.rnn(x, h0)

        out = self.fc(out)

        # print(out.shape)
        return out, hn


class RNNLM_bilstm(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM_bilstm, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.linear = nn.Linear(int(hidden_size * 2), vocab_size)

    def forward(self, x, h, epoch, out_inds):

        # Embed word ids to vectors
        x = self.embed(x)

        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))

        # print(out.shape)
        out = out[[i for i in out_inds]]

        # Decode hidden states of all time steps
        out = self.linear(out)

        return out, (h, c)
