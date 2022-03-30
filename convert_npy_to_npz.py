import numpy as np

models = [
    "bigram",
    "trigram",
    "rnn",
    "lstm",
    "bilstm",
    "bert",
    "bert_whole_word",
    "roberta",
    "xlm",
    "electra",
    "gpt2",
]
rng = np.load("prob_ranges_10_90.npy")
array_dict = {model: rng[i_model] for i_model, model in enumerate(models)}
np.savez("prob_ranges_10_90.npz", **array_dict)
