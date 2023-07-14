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

def get_vocabulary():
    return vocab_low, vocab_low_freqs, vocab_cap, vocab_cap_freqs

def get_token_controlled_vocabulary(models):
    """ returns a version of the vocabulary containing only words the have equal number tokens in all models specified
        args:
            models: list of models to be considered
        returns:
            vocab_low, vocab_low_freqs, vocab_cap, vocab_cap_freqs
"""

    def filter_vocab(vocab, models):
        """ filter a specific vocabulary
            args:
                vocab: the vocabulary to be filtered (list)
                models: the models to be considered (list)
            returns:
                filtered_vocab: the filtered vocabulary
        """

        filtered_vocab = vocab.copy()
        for word in vocab:
            token_counts = None
            for model in models:
                if token_counts is None:
                    token_counts = model.get_token_counts(word)
                else:
                    if token_counts != model.get_token_counts(word):
                        filtered_vocab.remove(word)
                        break
        return filtered_vocab

    return (
        filter_vocab(vocab_low,models),
        filter_vocab(vocab_low_freqs,models),
        filter_vocab(vocab_cap,models),
        filter_vocab(vocab_cap_freqs,models)
    )
