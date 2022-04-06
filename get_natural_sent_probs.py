# evaluate sentence probabilities of reddit sentences (needed for natural controversial sentence pair selection)
# this file should be run for each model

import numpy as np
import os
import argparse
import pathlib

from tqdm import tqdm

from model_functions import model_factory

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--gpu", default=0)

args = parser.parse_args()

model = args.model

model1 = model_factory(model, args.gpu)

file = open(
    os.path.join(
        "resources",
        "sentence_corpora",
        "natural_sentences_for_natural_controversial_sentence_pair_selection.txt",
    ),
    "r",
)
sents = file.read()
sents = sents.split("\n")

probs = []
for sent in tqdm(sents):
    prob = model1.sent_prob(sent)
    probs.append(prob)

probs = np.array(probs)

target_folder = os.path.join("resources", "precomputed_sentence_probabilities")
pathlib.path(target_folder).mkdir(parents=True, exist_ok=True)

np.save(
    os.path.join(
        target_folder,
        "natural_sentences_for_natural_controversial_sentence_pair_selection_probs_"
        + model
        + ".npy",
    ),
    probs,
)
