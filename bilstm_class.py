import torch.nn as nn


class RNNLM_bilstm(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
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
