import os

import torch

from lstm_class import RNNLM
from bilstm_class import RNNLM_bilstm
from rnn_class import RNNModel

from model_functions import model_factory

model_classes = [RNNLM, RNNLM_bilstm, RNNModel]
model_names = ["lstm", "bilstm", "rnn"]

for model_class, model_name in zip(model_classes, model_names):
    model = torch.load(f"contstim_{model_name}.pt")
    torch.save(
        model.state_dict(),
        os.path.join("model_checkpoints", model_name + "_state_dict.pt"),
    )

# testing

model = model_factory("rnn", 0)
print(model.sent_prob("The dog are running far away"))
print(model.sent_prob("The dog is running far away"))
print(model.sent_prob("The dogs are running far away"))
print(model.sent_prob("The dogs is running far away"))
