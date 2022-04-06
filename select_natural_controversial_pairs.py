# %% Setup and load data
import glob, os, pathlib
import itertools
import pickle
import argparse
from distutils.util import strtobool

import numpy as np
import scipy.stats
import pandas as pd
import gurobipy as gb  # Gurobi requires installing a license first (there's a free academic license)
from gurobipy import GRB

strbool = lambda x: bool(strtobool(str(x)))

default_txt_fname = os.path.join(
    "resources",
    "sentence_corpora",
    "natural_sentences_for_natural_controversial_sentence_pair_selection.txt",
)

parser = argparse.ArgumentParser()
parser.add_argument("--natural_sentences_file", type=str, default=default_txt_fname)
parser.add_argument(
    "--output_file",
    type=str,
    default=default_txt_fname.replace(".txt", "_selected.csv"),
)
parser.add_argument(
    "--precomputed_sentence_probabilities_folder",
    type=str,
    default="resources/precomputed_sentence_probabilities",
)
parser.add_argument("--allow_only_prepositions_to_repeat", type=strbool, default=True)
args = parser.parse_args()

npys = glob.glob(os.path.join(args.precomputed_sentence_probabilities_folder, "*.npy"))

models = [s.split(".")[-2].split("_")[-1] for s in npys]
n_models = len(models)

if n_models >= 1:
    print(f"found files for {n_models} models:", models)
else:
    print(
        "no precomputed sentence probability files found, run first get_natural_sent_probs.py for each model"
    )

with open(args.natural_sentences_file, "r") as f:
    sentences = f.readlines()
sentences = [s.strip() for s in sentences]
n_sentences = len(sentences)
print(f"loaded {n_sentences} sentences.")

log_prob = np.zeros((n_sentences, n_models))
log_prob[:] = np.nan

for i_model, (model, npy) in enumerate(zip(models, npys)):
    log_prob[:, i_model] = np.load(npy)

# %% rank sentence probabilities within each model
ranked_log_prob = scipy.stats.rankdata(log_prob, axis=0, method="average") / len(
    log_prob
)

# %% throw sentences that are not controversial at all
if args.allow_only_prepositions_to_repeat:
    allowed_repeating_words = set(
        pickle.load(open(os.path.join("resources", "preps.pkl"), "rb"))
    )
    keep_words_unique = True
else:
    allowed_repeating_words = None
    keep_words_unique = False


def prefilter_sentences(
    sentences: list,
    ranked_log_prob: np.ndarray,
    keep_words_unique=True,
    allowed_repeating_words=None,
):
    """prefilter sentences for controversiality according to at least one model pair

    keep_words_unique (bool): all words must be unique (within sentence)
    allowed_repeating_words (list): if keep_words_unique is True, these words are excluded from the uniqueness constrain. The words should be lower-case.

    """
    mask = np.zeros((n_sentences,), dtype=bool)  # True for sentence to be used.
    for i_model in range(n_models):
        for j_model in range(n_models):
            if i_model == j_model:
                continue

            p1 = ranked_log_prob[:, i_model]
            p2 = ranked_log_prob[:, j_model]

            cur_mask = np.logical_and(p1 < 0.5, p2 >= 0.5)

            mask = np.logical_or(mask, cur_mask)

    if keep_words_unique:
        if allowed_repeating_words is None:
            allowed_repeating_words = []

        for idx in np.flatnonzero(mask):
            sentence = sentences[idx]
            non_prep_words = [
                w.lower()
                for w in sentence.split()
                if not (w.lower() in allowed_repeating_words)
            ]
            is_there_a_repetition = len(non_prep_words) > len(set(non_prep_words))
            if is_there_a_repetition:
                mask[idx] = False

    return list(np.asarray(sentences)[mask]), ranked_log_prob[mask]


sentences, ranked_log_prob = prefilter_sentences(
    sentences,
    ranked_log_prob,
    keep_words_unique=keep_words_unique,
    allowed_repeating_words=allowed_repeating_words,
)

n_sentences = len(sentences)
print("prefiltered the sentences to a total of", n_sentences, "sentences.")

# # %%  Use Gurobipy to select controversial sentences
def select_sentences_Gurobi(
    sentences, models, ranked_log_prob, n_trials_per_pair=10, mode="minsum"
):
    """Select controversial sentence pairs using integer linear programming.
    This function selects pairs of sentences s1 and s2 for multiple model pairs m1 and m2
    to minimize sum_i[r(s1_i|m1_i)+r(s2_i|m2_i)] (i indexes trials)
    s.t.,
    r(s1_i|m2_i)>=0.5,
    r(s2_i|m1_i)>=0.5,
    and no sentence is used more than once.

    mode='minmax' aggregates trials by the maximum function instead of summation, but it results
    in less controversial sentences.
    """
    n_sentences = len(sentences)
    assert len(ranked_log_prob) == n_sentences

    n_models = ranked_log_prob.shape[1]
    assert len(models) == n_models

    assert mode == "minsum" or mode == "minmax"

    model_1 = []
    model_2 = []
    for i_model, j_model in itertools.combinations(range(n_models), 2):
        for i_trial in range(n_trials_per_pair):
            model_1.append(i_model)
            model_2.append(j_model)

    n_trials = len(model_1)

    m = gb.Model()

    X = m.addMVar((n_sentences, n_trials, 2), vtype=GRB.BINARY)

    if mode == "minsum":
        loss = 0
    elif mode == "minmax":
        loss = m.addMVar((1,))

    for i_trial in range(n_trials):

        # Evaluate the rank probability of the selected sentences by calculating the
        # dot-product between the vector of sentence rank probabilities and a binary
        # vector, all 0 except 1.0 for the selected sentence for i_trial.

        s1_m1 = (
            ranked_log_prob[:, model_1[i_trial]].T @ X[:, i_trial, 0]
        )  # the ranked probability of s1 according to model 1 (scalar)
        s1_m2 = ranked_log_prob[:, model_2[i_trial]].T @ X[:, i_trial, 0]
        s2_m1 = ranked_log_prob[:, model_1[i_trial]].T @ X[:, i_trial, 1]
        s2_m2 = ranked_log_prob[:, model_2[i_trial]].T @ X[:, i_trial, 1]

        # we want s1_m1 and s2_m2 to be small, and s1_m2 and s2_m1 to be big.
        if mode == "minsum":
            loss += s1_m1 + s2_m2
        elif mode == "minmax":  # we didn't use this
            m.addConstr(loss >= s1_m1)
            m.addConstr(loss >= s2_m2)

        m.addConstr(s1_m2 >= 0.5)
        m.addConstr(s2_m1 >= 0.5)

        # each trial should have one s1 and one s2.
        m.addConstr(X[:, i_trial, 0].sum() == 1)
        m.addConstr(X[:, i_trial, 1].sum() == 1)

    # no sentence should be used more than once.
    for i_sentence in range(n_sentences):
        m.addConstr(X[i_sentence].sum() <= 1)

    m.setObjective(loss, GRB.MINIMIZE)
    m.update()

    m.optimize()

    print("Obj: %g" % m.objVal)
    assert m.status == gb.GRB.Status.OPTIMAL

    solution = X.X
    # extracting solution
    df = []
    for i_trial in range(n_trials):
        s1_idx = np.flatnonzero(solution[:, i_trial, 0])[0]
        s2_idx = np.flatnonzero(solution[:, i_trial, 1])[0]
        d = {
            "sentence1": sentences[s1_idx],
            "sentence2": sentences[s2_idx],
            "model_1": models[model_1[i_trial]],
            "model_2": models[model_2[i_trial]],
        }
        d["s1_ranked_log_prob_model_1"] = ranked_log_prob[s1_idx, model_1[i_trial]]
        d["s1_ranked_log_prob_model_2"] = ranked_log_prob[s1_idx, model_2[i_trial]]
        d["s2_ranked_log_prob_model_1"] = ranked_log_prob[s2_idx, model_1[i_trial]]
        d["s2_ranked_log_prob_model_2"] = ranked_log_prob[s2_idx, model_2[i_trial]]
        df.append(d)
    df = pd.DataFrame(df)
    return df


print("solving ILP problem (this takes some time...)")
df = select_sentences_Gurobi(sentences, models, ranked_log_prob, mode="minsum")
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option("display.width", 1000)
print(df)

target_folder = pathlib.Path(os.path.dirname(args.output_file)).mkdir(
    parents=True, exist_ok=True
)
df.to_csv(args.output_file)
print("saved selected files in", args.output_file)
