# evaluate sentence probabilities of reddit sentences (needed for natural controversial sentence pair selection)
# this script should be run for each candidate model

import numpy as np
import os
import argparse
import pathlib

from tqdm import tqdm

from model_functions import model_factory

default_txt_fname = os.path.join(
    "resources",
    "sentence_corpora",
    "natural_sentences_for_natural_controversial_sentence_pair_selection.txt",
)

default_output_file = os.path.join(
    "resources",
    "precomputed_sentence_probabilities",
    "natural_sentences_for_natural_controversial_sentence_pair_selection_probs",
)

parser = argparse.ArgumentParser()
parser.add_argument("--natural_sentences_file", type=str, default=default_txt_fname)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--gpu", default=0)
parser.add_argument("--output_file", type=str, default=default_output_file)

args = parser.parse_args()

model = args.model

model1 = model_factory(model, args.gpu)

file = open(
    args.natural_sentences_file,
    "r",
)
sents = file.read()
sents = sents.split("\n")

probs = []
for sent in tqdm(sents):
    prob = model1.sent_prob(sent)
    probs.append(prob)

probs = np.array(probs)

target_folder = os.path.dirname(args.output_file)
pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True)

np.save(
    args.output_file + "_" + model + ".npy",
    probs,
)
