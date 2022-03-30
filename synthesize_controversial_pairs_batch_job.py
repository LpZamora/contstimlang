import os, pickle
import itertools, random
import random

import torch
import numpy as np
import pandas as pd

from model_functions import model_factory
from sentence_optimization import optimize_sentence_set, initialize_random_word_sentence
from utils import exclusive_write_line
from task_scheduler import TaskScheduler


class NaturalSentenceAssigner:
    """Assign natural sentences as initial sentences. Each natural sentence is assigned for one model pair."""

    def __init__(self, all_model_names, seed=42, natural_sentence_file=None):
        if natural_sentence_file is None:
            natural_sentence_file = os.path.join(
                "resources",
                "sentence_corpora",
                "natural_sentences_for_synthetic_controversial_sentence_pair_optimization.txt",
            )
        with open(natural_sentence_file) as f:
            natural_sentences = [l.strip().rstrip(".") for l in f]

        natural_sentences = pd.DataFrame({"sentence": natural_sentences})
        natural_sentences = natural_sentences.sample(
            frac=1, random_state=42
        )  # shuffle sentences

        model_pairs = list(itertools.combinations(all_model_names, 2))
        model_pairs = [tuple(sorted(pair)) for pair in model_pairs]

        random.Random(seed).shuffle(model_pairs)

        n_model_pairs = len(model_pairs)

        sentence_groups = natural_sentences.groupby(
            np.arange(len(natural_sentences)) % n_model_pairs
        )

        self.all_model_names = all_model_names
        self.model_pairs = model_pairs
        self.model_pair_dict = {
            tuple(model_pair): sentence_group[1].sort_index()
            for model_pair, sentence_group in zip(model_pairs, sentence_groups)
        }

    def get_sentences(self, model_pair):
        return self.model_pair_dict[tuple(sorted(model_pair))]


def synthesize_controversial_sentence_pair_set(
    all_model_names,
    initial_sentence_assigner,
    results_csv_folder=None,
    sent_len=8,
    allow_only_prepositions_to_repeat=True,
    max_sentence_pairs_per_run=5,
    natural_initialization=True,
    direction="down",
    n_pairs_to_synthesize_per_model_pair=100,
    verbose=3,
):
    """Synthesize a set of controversial synthetic sentence pairs.

    This function can be run in parallel by multiple nodes to build a large set of sentence pairs.

    args:
        all_model_names: list of strings, names of all models
        initial_sentence_assigner: NaturalSentenceAssigner object
        results_csv_folder: string, path to folder where the resulting sentence pairs will be saved
        sent_len: int, length of synthetic sentences (number of words)
        allow_only_prepositions_to_repeat: bool, if True, only prepositions can be repeated in the sentence
        max_pairs: int, maximum number of sentence pairs to synthesize with each run of the script (set to None to keep running). Useful if HPC jobs are time-limited.
        natural_initialization: bool, if True, use natural sentences as initial sentences. Otherwise, initialize as random sentences.
        direction: string, 'down' or 'up', direction of the sentence optimization. Only 'down' was used in the paper.
        n_pairs_to_synthesize_per_model_pair: int, number of sentence pairs to synthesize for each model pair.
        verbose: int, verbosity level.
    """

    n_sentences = 2  # we optimize a pair of sentences

    sentences_to_change = [1]  # change the second sentence

    sch = TaskScheduler(
        max_job_time_in_seconds=3600 * 6
    )  # tracking multiple sentence optimization jobs
    job_df = (
        sch.to_pandas()
    )  # get a dataframe of jobs, this DataFrame will be non-empty if there are jobs running

    try:
        job_id_df = pd.DataFrame(list(job_df["job_id"]))
    except:
        job_id_df = None

    # determine which model pair has the least completed or running jobs
    model_pairs_stats = []
    for model_name_pair in itertools.product(all_model_names, repeat=2):
        [model1_name, model2_name] = model_name_pair
        if model1_name == model2_name:
            continue
        if job_id_df is not None and len(job_id_df) > 0:
            n_jobs = (
                (job_id_df["model_1"] == model1_name)
                & (job_id_df["model_2"] == model2_name)
            ).sum()
        else:
            n_jobs = 0
        model_pairs_stats.append(
            {
                "model_1": model1_name,
                "model_2": model2_name,
                "n_jobs": n_jobs,
                "tie_breaker": random.random(),
            }
        )
    model_pairs_stats = pd.DataFrame(model_pairs_stats)
    model_pairs_stats = model_pairs_stats.sort_values(
        by=["n_jobs", "tie_breaker"], ascending=True
    )

    # list model pairs, pairs with less sentences first
    model_pair_list = list(
        zip(model_pairs_stats["model_1"], model_pairs_stats["model_2"])
    )

    if allow_only_prepositions_to_repeat:  # load a list of prepositions
        allowed_repeating_words = set(
            pickle.load(open(os.path.join("resources", "preps.pkl"), "rb"))
        )
        keep_words_unique = True
    else:
        allowed_repeating_words = None
        keep_words_unique = False

    for model_name_pair in model_pair_list:
        [model1_name, model2_name] = model_name_pair

        if results_csv_folder is not None:
            results_csv_fname = os.path.join(
                results_csv_folder, model1_name + "_vs_" + model2_name + ".csv"
            )
        external_stopping_check = lambda: False

        # allocate GPUs. Ideally, we'd like a separate GPU for each model.
        model_GPU_IDs = []
        cur_GPU_ID = 0
        for model_name in model_name_pair:
            model_GPU_IDs.append(cur_GPU_ID)
            if not model_name in [
                "bigram",
                "trigram",
            ]:  # bigram and trigram models run on CPU, so gpu_id will be ignored
                cur_GPU_ID += 1
                if cur_GPU_ID >= torch.cuda.device_count():
                    cur_GPU_ID = 0

        models_loaded = False

        n_optimized = 0

        natural_sentence_df = initial_sentence_assigner.get_sentences(model_name_pair)
        for i_natural_sentence, (sentence_index, natural_sentence) in enumerate(
            zip(natural_sentence_df.index, natural_sentence_df["sentence"])
        ):

            if i_natural_sentence >= (n_pairs_to_synthesize_per_model_pair + 1):
                break

            job_id = {
                "natural_sentence": natural_sentence,
                "model_1": model1_name,
                "model_2": model2_name,
            }

            success = sch.start_job(
                job_id
            )  # tracking the optimization job (useful for HPC environments)
            if not success:
                continue

            print(
                "optimizing sentence {} ({}) for {} vs {}".format(
                    i_natural_sentence, sentence_index, model1_name, model2_name
                )
            )
            if not models_loaded:  # load models
                models = []
                for model_name, model_GPU_ID in zip(model_name_pair, model_GPU_IDs):
                    print(
                        "loading "
                        + model_name
                        + " into gpu "
                        + str(model_GPU_ID)
                        + "...",
                        end="",
                    )
                    models.append(model_factory(model_name, model_GPU_ID))
                    print("done.")
                    models_loaded = True

            if direction == "down":

                def loss_func(sentences_log_p):
                    """
                    Given a reference natural sentence, form a variant of it which is at least as likely according to one model,
                    and as unlikely as possible according to the other.

                    args:
                    sentences_log_p (numpy.array, D_designs x M_models x S_sentences, or M_models x S_sentences) current log probabilities,
                    such that sentences_log_p[m,s]=log_p(sentence_s|model_m

                    comment: both M_models and S_sentences should be 2
                    """

                    # we'd like to push down the log-probability assigned by model 2 to s2 (the optimized sentence) as much as possible:
                    m2s1 = sentences_log_p[..., 1, 0]
                    m2s2 = sentences_log_p[..., 1, 1]
                    l = (
                        m2s2 - m2s1
                    )  # m2s1 is actually constant, we subtract it so l=0 for sentences with identical log-prob.

                    # this penalty is activated when model 1 assigns lower log-probability to s2 (the optimized sentence) compared to s1 (the reference sentence):
                    m1s1 = sentences_log_p[..., 0, 0]
                    m1s2 = sentences_log_p[..., 0, 1]
                    p = np.maximum(m1s1 - m1s2, 0.0)

                    return (
                        l + 1e5 * p
                    )  # we care more about satisfying the constraints than decreasing the loss

            elif direction == "up":

                def loss_func(sentences_log_p):
                    """
                    Given a reference low-probability sentence, form a variant of it which is as at least as unlikely according to one model,
                    and as likely as possible according to the other.

                    args:
                    sentences_log_p (numpy.array, D_designs x M_models x S_sentences, or M_models x S_sentences) current log probabilities,
                    such that sentences_log_p[m,s]=log_p(sentence_s|model_m

                    comment: both M_models and S_sentences should be 2

                    *** This approach was not used in the paper. ***
                    """

                    # we'd like to push up the log-probability assigned by model 2 to s2 (the optimized sentence) as much as possible:
                    m2s1 = sentences_log_p[..., 1, 0]
                    m2s2 = sentences_log_p[..., 1, 1]
                    l = (
                        m2s1 - m2s2
                    )  # m2s1 is actually constant, we add it so l=0 for sentences with identical log-prob.

                    # this penalty is activated when model 1 assigns higher log-probability to s2 (the optimized sentence) compared to s1 (the reference sentence):
                    m1s1 = sentences_log_p[..., 0, 0]
                    m1s2 = sentences_log_p[..., 0, 1]
                    p = np.maximum(m1s2 - m1s1, 0.0)

                    return (
                        l + 1e5 * p
                    )  # we care more about satisfying the constraints than decreasing the loss

            def monitoring_func(sentences, sentences_log_p):
                """prints an update on optimization status"""
                print(
                    model1_name
                    + ":"
                    + "{:.2f}/{:.2f}".format(
                        sentences_log_p[..., 0, 0], sentences_log_p[..., 0, 1]
                    )
                )
                print(
                    model2_name
                    + ":"
                    + "{:.2f}/{:.2f}".format(
                        sentences_log_p[..., 1, 0], sentences_log_p[..., 1, 1]
                    )
                )

            internal_stopping_condition = (
                lambda loss: False
            )  # stops optimization if condition is met

            if natural_initialization:
                initial_sentences = [natural_sentence] * n_sentences
            else:
                initial_sentences = [
                    initialize_random_word_sentence(
                        sent_len, initial_sampling="uniform"
                    )
                ] * n_sentences

            results = optimize_sentence_set(
                n_sentences,
                models=models,
                loss_func=loss_func,
                sentences=initial_sentences,
                sent_len=sent_len,
                initial_sampling="uniform",
                external_stopping_check=external_stopping_check,
                monitoring_func=monitoring_func,
                internal_stopping_condition=internal_stopping_condition,
                start_with_identical_sentences=True,
                max_steps=10000,
                keep_words_unique=keep_words_unique,
                allowed_repeating_words=allowed_repeating_words,
                sentences_to_change=sentences_to_change,
                verbose=verbose,
            )

            if results is False:  # optimization was terminated
                continue

            sentences = results["sentences"]
            sentences_log_p = results["sentences_log_p"]
            print(sentences)
            monitoring_func(sentences, sentences_log_p)

            # save results.
            # CSV format:
            # sentence 1, sentence 2, loss, model_1_log_prob_sent1, model_1_log_prob_sent2, model_2_log_prob_sent1, model_2_log_prob_sent2,
            outputs = (
                [sentence_index]
                + results["sentences"]
                + [results["loss"]]
                + list(sentences_log_p.flat)
            )
            line = ",".join(map(str, outputs))
            exclusive_write_line(results_csv_fname, line)
            sch.job_done(job_id, results=results)

            n_optimized += 1
            if (max_sentence_pairs_per_run is not None) and (
                n_optimized >= max_sentence_pairs_per_run
            ):
                break


if __name__ == "__main__":
    all_model_names = [
        "bigram",
        "trigram",
        "rnn",
        "lstm",
        "gpt2",
        "bert",
        "roberta",
        "xlm",
        "electra",
    ]

    initial_sentence_assigner = NaturalSentenceAssigner(all_model_names)
    sent_len = 8

    results_csv_folder = os.path.join(
        "synthesized_sentences",
        "controverisal_sentence_pairs_natural_initialization",
        "{}_word".format(sent_len),
    )

    synthesize_controversial_sentence_pair_set(
        all_model_names,
        initial_sentence_assigner,
        results_csv_folder=results_csv_folder,
        sent_len=sent_len,
        allow_only_prepositions_to_repeat=True,
        natural_initialization=True,
        direction="down",
        max_sentence_pairs_per_run=5,
        verbose=3,
    )
