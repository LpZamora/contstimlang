import os
import glob
import re

import pandas as pd
import numpy as np

import select_synthetic_controversial_sentences_for_behav_exp

csv_path = "/mnt/axon/nklab/projects/contstimlang/contstimlang/synthesized_sentences/bidirectional_prob_calc_exp/controverisal_sentence_pairs_natural_initialization/8_word"

raw_args = [
    "--synthesized_sentence_csv_folder",
    csv_path,
    "--output_csv_path",
    csv_path,
    "--n_sentence_triplets_for_consideration",
    "95",
    "--n_sentence_triplets_for_human_testing",
    "40",
    "--selection_method",
    "decile_balanced",
    "--enforce_natural_sentence_uniqueness",
]

select_synthetic_controversial_sentences_for_behav_exp.main(raw_args)
