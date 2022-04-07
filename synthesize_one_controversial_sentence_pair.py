import os, pickle
import argparse

import torch

from model_functions import model_factory
from sentence_optimization import optimize_sentence_set, controversiality_loss_func
from utils import exclusive_write_line


def synthesize_controversial_sentence_pair(
    model_accept,
    model_reject,
    initial_sentence,
    results_csv_fname=None,
    history_csv_fname=None,
    allow_only_prepositions_to_repeat=True,
    replacement_strategy="cyclic",
    verbose=3,
):
    """Synthesize a set of controversial synthetic sentence pairs.

    This function can be run in parallel by multiple nodes to build a large set of sentence pairs.

    args:
        all_model_names: list of strings, names of all models
        initial_sentence (str): the initial (natural) sentence to use for sentence initalization
        results_csv_fname: string, path to a file where the resulting sentence pair will be saved (it is appended to the file if it already exists)
        history_csv_fname: string, path to a file where the optimization history of sentence pairs will be saved
        allow_only_prepositions_to_repeat: bool, if True, only prepositions can be repeated in the sentence
        replacement_strategy: string, one of "cyclic" (as in the preprint,) "exhaustive" (try all possible replacements)
        verbose: int, verbosity level.

    if results_csv_fname is not None, generates a CSV file with the following columns:
    natural sentence, synthetic sentence, loss, log p(natural sentence | model1), log p(synthetic sentence | model1), log p(natural sentence | model2), log p(synthetic sentence | model2)
    """

    n_sentences = 2  # we optimize a pair of sentences

    sentences_to_change = [1]  # change the second sentence, keep the first fixed

    if allow_only_prepositions_to_repeat:  # load a list of prepositions
        allowed_repeating_words = set(
            pickle.load(open(os.path.join("resources", "preps.pkl"), "rb"))
        )
        keep_words_unique = True
    else:
        allowed_repeating_words = None
        keep_words_unique = False

    model_name_pair = [model_accept, model_reject]

    model1_name = model_accept
    model2_name = model_reject

    # allocate GPUs. Ideally, we'd like a separate GPU for each model.
    model_GPU_IDs = []
    cur_GPU_ID = 0
    for model_name in model_name_pair:
        if not model_name in [
            "bigram",
            "trigram",
        ]:  # bigram and trigram models run on CPU, so gpu_id will be ignored
            assert torch.cuda.device_count() > 0, "No GPU found"
            model_GPU_IDs.append(cur_GPU_ID)
            cur_GPU_ID += 1
            if cur_GPU_ID >= torch.cuda.device_count():
                cur_GPU_ID = 0
        else:
            model_GPU_IDs.append(0)
    models_loaded = False

    print(
        "optimizing sentence {} for {} vs {}".format(
            initial_sentence, model1_name, model2_name
        )
    )
    if not models_loaded:  # load models
        models = []
        for model_name, model_GPU_ID in zip(model_name_pair, model_GPU_IDs):

            device_type = "cpu" if model_name in ["bigram", "trigram"] else "gpu"
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
            models_loaded = True

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

    initial_sentence = initial_sentence.strip().strip(". ?!,")
    sent_len = len(initial_sentence.split())
    initial_sentences = [initial_sentence] * n_sentences

    save_history = history_csv_fname is not None

    results = optimize_sentence_set(
        n_sentences,
        models=models,
        loss_func=controversiality_loss_func,
        sentences=initial_sentences,
        sent_len=sent_len,
        initial_sampling=None,
        replacement_strategy=replacement_strategy,
        monitoring_func=monitoring_func,
        max_steps=10000,
        keep_words_unique=keep_words_unique,
        allowed_repeating_words=allowed_repeating_words,
        sentences_to_change=sentences_to_change,
        save_history=save_history,
        model_names=model_name_pair,
        verbose=verbose,
    )

    sentences = results["sentences"]
    sentences_log_p = results["sentences_log_p"]
    print(sentences)
    monitoring_func(sentences, sentences_log_p)

    if results_csv_fname is not None:
        # save results.
        # CSV format:
        # sentence 1, sentence 2, loss, model_1_log_prob_sent1, model_1_log_prob_sent2, model_2_log_prob_sent1, model_2_log_prob_sent2,
        outputs = results["sentences"] + [results["loss"]] + list(sentences_log_p.flat)
        line = ",".join(map(str, outputs))
        exclusive_write_line(results_csv_fname, line)

    if history_csv_fname is not None:
        results["history"].to_csv(history_csv_fname)


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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_accept",
        type=str,
        help="The model that will score the synthetic sentence to be at least as likely as the natural sentence",
        choices=all_model_names,
        default="bigram",
    )
    parser.add_argument(
        "--model_reject",
        type=str,
        help="The model that will score the synthetic sentence to be less likely than the natural sentence",
        choices=all_model_names,
        default="trigram",
    )
    parser.add_argument(
        "--initial_sentence",
        type=str,
        help="The initial sentence to use for sentence initalization. Use sentence case capitalization and no full stop (a period is added during model evaluation).",
        default="You can eat the cake and see the light",
    )
    parser.add_argument(
        "--results_csv_fname",
        type=str,
        help="The file to save the resulting sentence pair to. If the file already exists, the sentence pair is appended to the file.",
    )

    parser.add_argument(
        "--history_csv_fname",
        type=str,
        help="The file to save the optimization history to (optional).",
    )

    args = parser.parse_args()

    synthesize_controversial_sentence_pair(
        model_accept=args.model_accept,
        model_reject=args.model_reject,
        initial_sentence=args.initial_sentence,
        results_csv_fname=args.results_csv_fname,
        history_csv_fname=args.history_csv_fname,
    )
