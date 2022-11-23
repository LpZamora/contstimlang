import os, pickle
import random
import time

import torch
import numpy as np
import pandas as pd

from model_functions import model_factory
from sentence_optimization import (
    optimize_sentence_set,
    initialize_random_word_sentence,
    controversiality_loss_func,
)
from utils import exclusive_write_line
from task_scheduler import TaskScheduler
from batch_synthesize_controversial_pairs import NaturalSentenceAssigner
from absolute_acceptability_exp import import_Lau_2020_data, load_rating_model, acquire_absolute_rating_controversiality_fun


class Lau2020NaturalSentenceAssigner:
    """Assign natural sentences from Lau et al 2020 as initial sentences. Each model pair uses all of the sentences"""

    def __init__(self, seed=42):
        """Initialize the assigner.

        Args:            
            seed: random seed            
        """

        Lau_df = import_Lau_2020_data(context_condition="none",max_sentence_length=None)
        Lau_df = Lau_df[Lau_df["translated"]==0]
        self.natural_sentences = Lau_df.loc[:,["sentence"]]
        # shuffle the sentences
        random.seed(seed)
        # shuffle dataframe rows
        self.natural_sentences = self.natural_sentences.sample(frac=1, random_state=seed)

    def get_sentences(self, model_pair):
        return self.natural_sentences


def synthesize_absolute_rating_controversial_sentence_set(
    model_pairs,
    initial_sentence_assigner,
    results_csv_folder=None,
    sent_len=8,
    allow_only_prepositions_to_repeat=True,
    max_sentence_pairs_per_run=5,
    natural_initialization=True,
    replacement_strategy="cyclic",
    n_pairs_to_synthesize_per_model_pair=100,
    max_non_decreasing_loss_attempts_per_word=5,
    max_replacement_attempts_per_word=50,
    max_opt_hours=None,
    verbose=3,
):
    """Synthesize a set of controversial synthetic sentence pairs.

    This function can be run in parallel by multiple nodes to build a large set of sentence pairs.

    args:
        model_pairs: list of tuples of model names
        initial_sentence_assigner: NaturalSentenceAssigner object
        results_csv_folder: string, path to folder where the resulting sentence pairs will be saved
        sent_len: int, length of synthetic sentences (number of words)
        allow_only_prepositions_to_repeat: bool, if True, only prepositions can be repeated in the sentence
        max_pairs: int, maximum number of sentence pairs to synthesize with each run of the script (set to None to keep running). Useful if HPC jobs are time-limited.
        natural_initialization: bool, if True, use natural sentences as initial sentences. Otherwise, initialize as random sentences.
        n_pairs_to_synthesize_per_model_pair: int, number of sentence pairs to synthesize for each model pair.
        max_opt_hours: int, maximum number of hours to run the optimization for each sentence pair.
        verbose: int, verbosity level.

    Generates a CSV file with the following columns:
    natural sentence, synthetic sentence, loss, log p(natural sentence | model1), log p(synthetic sentence | model1), log p(natural sentence | model2), log p(synthetic sentence | model2)
    # model1 and model2 are the two models that are used to generate the sentence pair, as determined from the filename {model1}_vs_{model2}.
    """

    n_sentences = 1  # we optimize one sentence at a time

    sentences_to_change = [0]  

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
    print("job_id_df", job_id_df)

    # determine which model pair has the least completed or running jobs
    model_pairs_stats = []
    for model_name_pair in model_pairs:
        [model_accept_name, model_reject_name] = model_name_pair
        if model_accept_name == model_reject_name:
            continue
        if job_id_df is not None and len(job_id_df) > 0:
            n_jobs = (
                (job_id_df["model_accept"] == model_accept_name)
                & (job_id_df["model_reject"] == model_reject_name)
            ).sum()
        else:
            n_jobs = 0
        model_pairs_stats.append(
            {
                "model_accept": model_accept_name,
                "model_reject": model_reject_name,
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
        zip(model_pairs_stats["model_accept"], model_pairs_stats["model_reject"])
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
        [model_accept_name, model_reject_name] = model_name_pair

        if results_csv_folder is not None:
            results_csv_fname = os.path.join(
                results_csv_folder, model_accept_name + "_vs_" + model_reject_name + ".csv"
            )

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
                "model_accept": model_accept_name,
                "model_reject": model_reject_name,
            }

            success = sch.start_job(
                job_id
            )  # tracking the optimization job (useful for HPC environments)
            if not success:
                continue

            print(
                "optimizing sentence {} ({}) for {} vs {}".format(
                    i_natural_sentence, sentence_index, model_accept_name, model_reject_name
                )
            )
            if not models_loaded:  # load models
                models = []
                rating_modules = []
                for model_name, model_GPU_ID in zip(model_name_pair, model_GPU_IDs):
                    device_type = (
                        "cpu" if model_name in ["bigram", "trigram"] else "gpu"
                    )
                    print(
                        "loading "
                        + model_name
                        + " "
                        + device_type
                        + " "
                        + str(model_GPU_ID)
                        + "...",
                        end="",
                    )
                    models.append(model_factory(model_name, model_GPU_ID))
                    print("done.")
                                        
                    # load rating modules. These are used to map log-probabilities to 1-4 acceptability ratings.
                    rating_modules.append(
                        load_rating_model(model_name)
                    )
                    
                    models_loaded = True

            model_accept_rating_module, model_reject_rating_module = tuple(rating_modules)
            # TODO: replace this with the rating-based controversiality measure
            # get the controversiality function
            absolute_rating_controversiality = acquire_absolute_rating_controversiality_fun(
                model_accept_rating_module,
                model_reject_rating_module)
            
            def loss_func(*args, **kwargs):
                return -absolute_rating_controversiality(*args, **kwargs)

            def monitoring_func(sentences, sentences_log_p):
                print('\n')
                print(f"{model_accept_name} ratings:", model_accept_rating_module(
                    torch.tensor(sentences_log_p[0], dtype=torch.float32),
                    torch.tensor([len(sentences[0])], dtype=torch.float32)).item(),
                    f"log_p: {sentences_log_p[0]}")            
                    
                print(f"{model_reject_name} ratings:", model_reject_rating_module(
                    torch.tensor(sentences_log_p[1], dtype=torch.float32),
                    torch.tensor([len(sentences[0])], dtype=torch.float32)).item(),
                    f"log_p: {sentences_log_p[1]}")
            
            
            if max_opt_hours is not None:
                # stop optimization after max_opt_time hours
                start_time = time.time()

                def stop_if_time_exceeded(loss):
                    time_elapsed_in_hours = (time.time() - start_time) / 3600
                    if time_elapsed_in_hours > max_opt_hours:
                        print(
                            f"time exceeded ({time_elapsed_in_hours:.2f} hours), stopping optimization"
                        )
                        return True
                    else:
                        return False

                internal_stopping_condition = stop_if_time_exceeded
            else:
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

            sentence_length_in_words = len(initial_sentences[0].split())

            model_accept, model_reject = tuple(models)

            results = optimize_sentence_set(
                n_sentences=1,
                models=[model_accept, model_reject],
                loss_func=loss_func,
                sentences=initial_sentences,
                sent_length_in_words=sentence_length_in_words,                
                replacement_strategy="exhaustive",
                monitoring_func=monitoring_func,                
                max_steps=10000,
                internal_stopping_condition=internal_stopping_condition,                
                external_stopping_check=lambda: False,
                keep_words_unique=keep_words_unique,
                allowed_repeating_words=allowed_repeating_words,
                sentences_to_change=None,
                max_replacement_attempts_per_word=max_replacement_attempts_per_word,
                max_non_decreasing_loss_attempts_per_word=max_non_decreasing_loss_attempts_per_word,
                model_names=[model_accept_name, model_reject_name],
                do_pass_n_characters=True,
                verbose=verbose,
                save_history=True,
            )
            
            if results is False:  # optimization was terminated
                continue

            sentences = results["sentences"]
            sentences_log_p = results["sentences_log_p"]
            print(sentences)
            monitoring_func(sentences, sentences_log_p)

            # save results.
            # CSV format:
            # natural sentence, synthesized sentence, loss, model_1_log_prob_sent1, model_2_log_prob_sent1
            outputs = (
                [sentence_index]
                + [natural_sentence]
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
        "gpt2",
        "bert_new_implementation",
        "roberta_new_implementation",
        "electra_new_implementation",
    ]

    # get all pairs excluding self-pairs:
    model_pairs = []
    for model1_name in all_model_names:
        for model2_name in all_model_names:
            if model1_name != model2_name:
                model_pairs.append((model1_name, model2_name))
    
    initial_sentence_assigner = Lau2020NaturalSentenceAssigner(seed=42)

    results_csv_folder = os.path.join(
        "synthesized_sentences_test",
        "absolute_rating_controverisal_sentences_natural_initialization",        
    )

    synthesize_absolute_rating_controversial_sentence_set(
        model_pairs,
        initial_sentence_assigner,
        results_csv_folder=results_csv_folder,
        sent_len=None,  # Use the length of the initial sentence
        allow_only_prepositions_to_repeat=True,  # in the preprint, this was True
        natural_initialization=True,  # sample random sentence for initialization
        max_sentence_pairs_per_run=1,  # set this to a small number (e.g. 5) if HPC job time is limited, None if you want the code to keep running until it's done
        verbose=3,
    )
