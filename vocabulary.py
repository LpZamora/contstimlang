import pickle
import os

folder = os.path.join("resources", "vocabulary")
# load vocabulary and word probabilities
with open(os.path.join(folder, "vocab_low.pkl"), "rb") as file:
    vocab_low = pickle.load(file)

with open(os.path.join(folder, "vocab_low_freqs.pkl"), "rb") as file:
    vocab_low_freqs = pickle.load(file)

with open(os.path.join(folder, "vocab_cap.pkl"), "rb") as file:
    vocab_cap = pickle.load(file)

with open(os.path.join(folder, "vocab_cap_freqs.pkl"), "rb") as file:
    vocab_cap_freqs = pickle.load(file)
