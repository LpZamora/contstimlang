import os, pickle
import itertools, random

import torch
import numpy as np
import scipy.stats
from scipy.special import logsumexp

from model_functions import model_factory
from sentence_optimization import optimize_sentence_set
from .model_to_human_decision_torch import load_decision_model
from utils import exclusive_write_line, get_n_lines

def controversiality_score(decision_p, model_prior=None):
    """return the mutual information between model identities and predicted sentence choices for a single trial.

    args:
    decision_p (numpy.array) (D_designs, M_models, S_sentences) tensor such that decision_p[d,m,s] equals to log-p(sentence_s | model_m, design d).
    model_prior (numpy.array) M_models long vector of model prior probabilities

    returns:
    score (numpy.array) D_designs-long vector of scores.
    """

    D_designs, M_models, S_sentences = decision_p.shape

    if model_prior is None:
        model_prior = np.ones(shape=(M_models)) / M_models
    model_prior = model_prior.reshape((1, M_models, 1))

    # I(X,Y)=D_KL(P(X,Y) || P(X)*P(Y) )

    joint = decision_p * model_prior  # joint[d,m,s]=p(sentence_s, model_m | design_d)
    sentence_marginal = joint.sum(
        axis=1, keepdims=True
    )  # summate the joint over model to produce sentence_marginal[d,0,s]=p(sentence_s | design_d)

    marginal_product = (
        sentence_marginal * model_prior
    )  # marginal_product[d,m,s]=p(sentence_s | design_d)*p(model_m)

    MI = scipy.stats.entropy(
        pk=joint.reshape((D_designs, -1)),
        qk=marginal_product.reshape((D_designs, -1)),
        base=2,
        axis=1,
    )  # MI[d]=D_KL(joint[d],marginal_product[d])
    return MI


# log_DKL calculation
def log_DKL(log_p, log_q, axis=None):
    """calculate log(KL divergence(p||q)) where p and q are given in log scales.
    This function assumes that the probability distributions are normalized.

    D_KL(p||q)= sum(p*(log_p-log_q))
    log_D_KL(p||q)=logsum(p*(log_p-log_q)=
    logsum( (log_p-log_q) * exp(log_p) )=
    logsum( b * exp(a) )
    where b=log_p-log_q and a = log_p

    """
    b = log_p - log_q
    a = log_p

    mask = log_p == -np.inf
    a[mask] = 0
    b[mask] = 0

    return logsumexp(a, axis=axis, b=b, keepdims=False)


def controversiality_score_log_scale(decision_log_p, model_prior=None):
    """return the mutual information between model identities and predicted sentence choices for a single trial.

    args:
    decision_p (numpy.array) (D_designs, M_models, S_sentences) tensor such that decision_p[d,m,s] equals to log-p(sentence_s | model_m, design d).
    model_prior (numpy.array) M_models long vector of model prior probabilities

    returns:
    score (numpy.array) D_designs-long vector of scores.
    """

    D_designs, M_models, S_sentences = decision_log_p.shape

    if model_prior is None:
        model_prior = np.ones(shape=(M_models)) / M_models
    model_prior = model_prior.reshape((1, M_models, 1))
    log_model_prior = np.log(model_prior)

    # I(X,Y)=D_KL(P(X,Y) || P(X)*P(Y) )

    log_joint = (
        decision_log_p + log_model_prior
    )  # joint[d,m,s]=p(sentence_s, model_m | design_d)
    log_sentence_marginal = logsumexp(
        log_joint, axis=1, keepdims=True
    )  # summate the joint over model to produce sentence_marginal[d,0,s]=p(sentence_s | design_d)

    log_marginal_product = (
        log_sentence_marginal + log_model_prior
    )  # marginal_product[d,m,s]=p(sentence_s | design_d)*p(model_m)
    log_MI = log_DKL(
        log_joint.reshape((D_designs, -1)),
        log_marginal_product.reshape((D_designs, -1)),
        axis=1,
    )  # MI[d]=D_KL(joint[d],marginal_product[d])
    return log_MI


def human_choice_probability(
    sentences_log_p,
    human_choice_response_models=None,
    log_scale=False,
    grad_enabled=False,
):
    """calculate the probability of human choice of a sentence given model sentence-log-probabilities and human_choice_response_models

    args:
    sentences_log_p (numpy.array, D_designs x M_models x S_sentences) current log probabilities, such that sentences_log_p[m,s]=log_p(sentence_s|model_m)
    human_choice_response_models (None/a list of ModelToHumanDecision instances)
    grads_required (boolean)
    log_scale (boolean) if True, returns log_p instead of p

    returns:
        decision_p (numpy.array, D_designs x M_models x S_sentences) p(d,m,s)=the probability of human choice of sentence s given model m and design d
    """

    if sentences_log_p.ndim == 2:
        sentences_log_p = np.expand_dims(sentences_log_p, 0)

    D_designs, M_models, S_sentences = sentences_log_p.shape
    if (
        human_choice_response_models is None
    ):  # direct readout of choice probabilities (i.e., no squashing, gamma = 1.0)
        decision_log_p = sentences_log_p - logsumexp(
            sentences_log_p, axis=2, keepdims=True
        )  # normalize sentence probability by total sentence probability (within model and design)
    elif isinstance(human_choice_response_models, list):
        # model-specific human response model object
        assert (
            S_sentences == 2
        ), "ModelToHumanDecision currently assumes only two sentences"
        assert len(human_choice_response_models) == M_models

        decision_log_p = np.empty_like(sentences_log_p)
        with torch.set_grad_enabled(grad_enabled):
            for i_model in range(M_models):
                decision_log_p[:, i_model, :] = (
                    -human_choice_response_models[i_model](
                        sentences_log_p[:, i_model, :]
                    )
                    .cpu()
                    .numpy()
                )  # the minus is required because the models return NLL
    else:
        raise ValueError
    if log_scale:
        return decision_log_p
    else:
        decision_p = np.exp(decision_log_p)
        return decision_p


def human_choice_controversiality_loss(
    sentences_log_p, human_choice_response_models=None, model_prior=None
):
    """calculate model identification loss given sentence-log-probabilities and human_choice_response_models

    args:
    sentences_log_p (numpy.array, D_designs x M_models x S_sentences) current log probabilities, such that sentences_log_p[m,s]=log_p(sentence_s|model_m)
    human_choice_response_models (None/a list of ModelToHumanDecision instances)
    model_prior (numpy.array) M_models long vector of model prior probabilities

    returns:
    score (numpy.array) D_designs-long vector of scores.

    """
    decision_p = human_choice_probability(
        sentences_log_p,
        human_choice_response_models=human_choice_response_models,
        log_scale=False,
    )
    MI = controversiality_score(decision_p, model_prior=model_prior)
    loss = -MI
    return loss


def human_choice_controversiality_loss_log_scale(
    sentences_log_p, human_choice_response_models=None, model_prior=None
):
    """calculate model identification loss given sentence-log-probabilities and human_choice_response_models

    args:
    sentences_log_p (numpy.array, D_designs x M_models x S_sentences) current log probabilities, such that sentences_log_p[m,s]=log_p(sentence_s|model_m)
    human_choice_response_models (None/a list of ModelToHumanDecision instances)
    model_prior (numpy.array) M_models long vector of model prior probabilities

    returns:
    score (numpy.array) D_designs-long vector of scores.

    """
    decision_log_p = human_choice_probability(
        sentences_log_p,
        human_choice_response_models=human_choice_response_models,
        log_scale=True,
    )
    MI = controversiality_score_log_scale(decision_log_p, model_prior=model_prior)
    loss = -MI
    return loss


def synthesize_controversial_sentence_pair(
    all_model_names,
    decision_models_folder,
    results_csv_folder=None,
    sent_len=8,
    max_pairs=10,
    allow_only_prepositions_to_repeat=False,
    natural_initialization=False,
    sentences_to_change=None,
    verbose=3,
):
    n_sentences = 2  # we optimize a pair of sentences

    if allow_only_prepositions_to_repeat:
        allowed_repeating_words = set(pickle.load(open("preps.pkl", "rb")))
        keep_words_unique = True
    else:
        allowed_repeating_words = None
        keep_words_unique = False

    if natural_initialization:
        with open("sents10k.txt") as f:
            natural_sentences = [l.strip().rstrip(".") for l in f]

    for model_name_pair in itertools.combinations(all_model_names, 2):
        [model1_name, model2_name] = model_name_pair

        if results_csv_folder is not None:
            results_csv_fname = os.path.join(
                results_csv_folder, model1_name + "_vs_" + model2_name + ".csv"
            )

            # This function halts the sentence optimization if the file while its running
            def is_file_complete():
                is_complete = get_n_lines(results_csv_fname) >= max_pairs
                if is_complete and verbose >= 3:
                    print(
                        "found "
                        + str(get_n_lines(results_csv_fname))
                        + " lines in "
                        + results_csv_fname
                    )
                return is_complete

            if is_file_complete():  # check right away
                continue

            external_stopping_check = is_file_complete  # the check will be used within the optimization function
        else:  # no results saving, just display.
            external_stopping_check = lambda: False

        # allocate GPUs
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

        # load models
        models = []
        for model_name, model_GPU_ID in zip(model_name_pair, model_GPU_IDs):
            print(
                "loading " + model_name + " into gpu " + str(model_GPU_ID) + "...",
                end="",
            )
            models.append(model_factory(model_name, model_GPU_ID))
            print("done.")

        # load human decision models
        optimizer = "LBFGS"
        human_choice_response_models = []
        for model_name in model_name_pair:
            path = os.path.join(decision_models_folder, model_name + ".pkl")
            human_choice_response_models.append(load_decision_model(path, device="cpu"))

        model_prior = None  # assume flat prior over models for MI calculation

        while not is_file_complete():

            def loss_func(sentences_log_p):
                return human_choice_controversiality_loss_log_scale(
                    sentences_log_p,
                    human_choice_response_models=human_choice_response_models,
                    model_prior=model_prior,
                )

            def monitoring_func(sentences, sentences_log_p):
                human_p = human_choice_probability(
                    sentences_log_p,
                    human_choice_response_models=human_choice_response_models,
                    log_scale=False,
                )
                print(
                    model1_name
                    + ":"
                    + "{:.2f}/{:.2f}".format(human_p[0, 0, 0], human_p[0, 0, 1])
                )
                print(
                    model2_name
                    + ":"
                    + "{:.2f}/{:.2f}".format(human_p[0, 1, 0], human_p[0, 1, 1])
                )

            internal_stopping_condition = (
                lambda loss: False
            )  # don't stop optimizing until convergence

            if natural_initialization:
                initial_sentences = [random.choice(natural_sentences)] * n_sentences
            else:
                initial_sentences = None

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

            # write results to file:
            human_p = human_choice_probability(
                sentences_log_p,
                human_choice_response_models=human_choice_response_models,
                log_scale=False,
            )
            MI = controversiality_score(human_p)

            if not np.isclose(MI, 0):
                # sentence 1, sentence 2, MI, model_1_log_prob_sent1, model_1_log_prob_sent2, model_2_log_prob_sent1, model_2_log_prob_sent2,
                # model_1_human_prob_sent1, model_1_human_prob_sent2, model_2_human_prob_sent1, model_2_human_prob_sent2
                outputs = (
                    results["sentences"]
                    + [MI]
                    + list(sentences_log_p.flat)
                    + list(human_p.flat)
                )
                line = ",".join(map(str, outputs))
                exclusive_write_line(results_csv_fname, line, max_pairs)
            else:
                print("MI=0, not writing result.")


if __name__ == "__main__":
    # all_model_names=['bigram','trigram','rnn','lstm','gpt2','bert','bert_whole_word','roberta','xlm','electra','bilstm']
    all_model_names = ["bigram", "trigram", "rnn", "lstm", "gpt2"]
    # all_model_names=['bigram','trigram','gpt2']
    sent_len = 8

    optimizer = "LBFGS"
    decision_model_class = "FixedWidthSquashing"
    decision_models_folder = os.path.join(
        "decision_models",
        "20210115",
        decision_model_class + "_" + optimizer + "_{}_word".format(sent_len),
    )

    results_csv_folder = os.path.join(
        "synthesized_sentences",
        "20210115_controverisal_sentence_pairs_no_reps_natural_frozen_ref",
        decision_model_class + "_" + optimizer + "_{}_word".format(sent_len),
    )

    synthesize_controversial_sentence_pair(
        all_model_names,
        decision_models_folder,
        results_csv_folder=results_csv_folder,
        sent_len=sent_len,
        allow_only_prepositions_to_repeat=True,
        natural_initialization=True,
        max_pairs=10,
        verbose=3,
        sentences_to_change=[
            0,
        ],
    )
