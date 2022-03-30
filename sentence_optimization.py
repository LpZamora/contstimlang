#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import random
import math
import pathlib

import numpy as np
import portalocker
import scipy.stats
from scipy.special import logsumexp
import pandas as pd
import torch

from model_functions import model_factory
from interpolation_search import SetInterpolationSearchPandas
from model_to_human_decision_torch import ModelToHumanDecision

from vocabulary import vocab_low, vocab_low_freqs, vocab_cap, vocab_cap_freqs


def initialize_random_word_sentence(sent_len, initial_sampling="uniform"):
    if initial_sampling == "uniform":
        vocab_low_freqs1 = np.ones([len(vocab_low_freqs)]) / len(vocab_low_freqs)
        vocab_cap_freqs1 = np.ones([len(vocab_cap_freqs)]) / len(vocab_cap_freqs)
    elif initial_sampling == "proportional":
        vocab_low_freqs1 = vocab_low_freqs
        vocab_cap_freqs1 = vocab_cap_freqs
    else:
        raise ValueError("unsupported initial_sampling argument")

    first_word = np.random.choice(vocab_cap, 1, p=vocab_cap_freqs1)
    words = list(first_word) + list(
        np.random.choice(vocab_low, sent_len - 1, p=vocab_low_freqs1, replace=False)
    )

    sent = " ".join(words)

    return sent


def external_stopping_checking():
    if get_n_lines(fname) >= max_sentences:
        if verbose >= 3:
            print("found " + str(get_n_lines(fname)) + " lines in " + fname)
        return True
    else:
        return False


def log_likelihood_target_stopping_condition(loss):
    if np.abs(loss) < 1:
        print("target log-prob achieved, sentence added to file.")
        return True
    else:
        return False

        # exclusive_write_line(fname,sent1+'.',max_lines=max_sentences)


def get_word_prob_from_model(model, words, wordi):
    """evaluate the log-probability of sentences resulting from replacing the word at location wordi in the sentence defined by 'words'.
    words (list of strings)
    wordi (int) (0 for first word and so on).

    returns:
    word list (list)
    model_word_probs (np.Array) exact or approximate log-probability of each sentence resulting from replacing the word at wordi with a word from word_list
    prob_type (str) either 'exact' or 'approximate'

    #TODO: move this into the models' word_prob methods.
    """

    if wordi == 0:
        vocab = vocab_cap
    else:
        vocab = vocab_low

    output = model.word_probs(
        words.copy(), wordi
    )  # copying ensures words remains unchanged regardless of the model code
    if len(output) == 2 and isinstance(
        output, tuple
    ):  # model returns indecis and probabilities
        model_word_inds = output[1]
        model_word_probs = output[0]
        word_list = [vocab[w] for w in model_word_inds]
    else:  # probabilities are returned for all vocab
        model_word_probs = output
        assert len(model_word_probs) == len(vocab)
        word_list = list(vocab.copy())

    return word_list, model_word_probs


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


def get_loss_fun_with_respect_to_sentence_i(sentences_log_p, i_sentence, loss_func):
    """return a loss function with respect to updating the model probabilities of just one of the sentences (the i_sentence-th).
    This help function simplifies loss-evaluation when we seek to update one of the sentences.

    args:
    sentences_log_p (numpy.array , M_models x S_sentences) current log probabilities, such that sentences_log_p[m,s]=log_p(sentence_s|model_m)
    i_sentence (int) which sentence are we considering to change
    loss_func (function) a function that maps design-model-sentence log-probabilities tensor (D_designs x M_models x S_sentence) to D_designs-long loss vector

    # returns:
    loss_func_i (function) a function that maps D_designs-long vector of sentence log-probabilities to D_designs-long loss vector
    """

    def loss_func_i(sentence_i_log_p):
        """return loss as function of updating sentence probabilities of sentence_i in a sentences_log_p matrix (D_designs x M_models)
        args:
            sentence_i_log_p (numpy.array) M_models x S_sentences
        returns:
            loss (numpy.array) M_sentences
        """

        D_designs, M_models = sentence_i_log_p.shape

        # copy the sentence_log_p matrix M_designs times to form an D_designs x M_models x S_sentences tensor
        modified_sentences_log_p = np.repeat(
            np.expand_dims(sentences_log_p, 0), D_designs, axis=0
        )

        # paste sentence_i_log_p (log probabilities of potential replacement sentences for each model)
        modified_sentences_log_p[:, :, i_sentence] = sentence_i_log_p

        loss = loss_func(modified_sentences_log_p)
        return loss

    return loss_func_i


def display_sentences_with_highlighted_word(sentences, i_sentence, wordi):
    """display sentences with the wordi-ith word in the i_sentence highlighted"""
    for j_sentence, sentence in enumerate(sentences):
        if j_sentence == i_sentence:
            words = sentence.split(" ")
            words[wordi] = "*" + words[wordi].upper() + "*"
            modified_sentence = " ".join(words)
            print(modified_sentence, end="")
        else:
            print(sentence, end="")
        if j_sentence == 0:
            print("/", end="")
        else:
            print("")


def optimize_sentence_set(
    n_sentences,
    models,
    loss_func,
    sent_len=8,
    sentences=None,
    initial_sampling="uniform",
    external_stopping_check=lambda: False,
    internal_stopping_condition=lambda: False,
    start_with_identical_sentences=True,
    max_steps=10000,
    monitoring_func=None,
    max_replacement_attempts_per_word=50,
    max_non_decreasing_loss_attempts_per_word=5,
    keep_words_unique=False,
    allowed_repeating_words=None,
    sentences_to_change=None,
    verbose=3,
):
    """Optimize a sentence set of n_sentences.
    n_sentences (int) how many sentences to optimize (e.g., 2 for a sentence pair).
    models (list) a list of model objects
    loss_func (function) given a [d,ms] log_p(sentence s|model m, d d)
    sentence_len (int) how many words
    sentences (list) initial sentences (if None, sentences are randomly sampled).
    initial_sampling (str) 'uniform' or 'proportional'
    start_with_identical_sentences (boolean)
    returns a results dict, or False (if aborted due to external_stopping_check)
    max_steps (int) maximum total number of word replacements
    monitoring_func (function) function for online user feedback. args: (sentences, sentences_log_p)
    max_replacement_attempts_per_word (int) quit trying to replace a word after max_replacement_attempts_per_word alternative sentences were evaluated
    max_non_decreasing_loss_attempts_per_word (int) quit trying to replace a word after max_non_decreasing_loss_attempts_per_word sentences did not show decreasing loss
    keep_words_unique (bool) all words must be unique (within sentence)
    allowed_repeating_words (list) if keep_words_unique is True, these words are excluded from the uniqueness constrain. The words should be lower-case.
    sentences_to_change (list) which sentences should be optimized (e.g., [0,1]). Default: all sentences.
    verbose (int)
    """

    # initializes a list (wordis) that determines which word is replaced at each step.
    # the order is designed to be cyclical
    def get_word_replacement_order(sent_len, max_steps):
        wordi = np.arange(sent_len)
        wordis = []
        while len(wordis) < max_steps:
            random.shuffle(wordi)
            wordis = wordis + list(wordi)
        return wordis

    wordis = get_word_replacement_order(sent_len, max_steps)

    if sentences is not None:
        # sentences provided. do same sanity checks.
        n_sentences = len(sentences)
        sentence_len = np.unique([len(sentence.split(" ")) for sentence in sentences])
        assert len(sentence_len) == 1, "all sentences should be of the same length"
        sentence_len = sentence_len.item()
    else:  # initialize random sentences
        if start_with_identical_sentences:
            sentences = [
                initialize_sentence(sent_len, initial_sampling=initial_sampling)
            ] * n_sentences
        else:
            sentences = [
                initialize_sentence(sent_len, initial_sampling=initial_sampling)
                for i_sent in range(n_sentences)
            ]

    if sentences_to_change is None:
        sentences_to_change = list(
            range(n_sentences)
        )  # by default change all sentences

    # get initial sentence probabilities
    def get_sentence_log_probabilities(models, sentences):
        """Return a (model x sentence) log_probability numpy matrix"""
        log_p = np.empty(shape=(len(models), len(sentences)))
        log_p[:] = np.nan
        for i_model, model in enumerate(models):
            for i_sentence, sentence in enumerate(sentences):
                log_p[i_model, i_sentence] = model.sent_prob(sentence)
        return log_p

    # sentence_log_p[m,s] is the log-probabilitiy assigned to sentence s by model m.
    sentences_log_p = get_sentence_log_probabilities(models, sentences)
    current_loss = loss_func(sentences_log_p).item()

    if verbose >= 2:
        print("initialized:")
        for sentence in sentences:
            print(sentence)
        print("loss:", current_loss)

    n_consequetive_failed_replacements = 0  # (previsouly 'cycle' - this keeps track on how many words we failed to replace since the last succesful word replacement)
    termination_reason = ""
    for step in range(max_steps):
        if external_stopping_check():
            # abort optimization (e.g., another worker completed the sentence).
            return False

        if internal_stopping_condition(current_loss):
            termination_reason = "internal_stopping_condition"
            break
        elif n_consequetive_failed_replacements == sent_len:
            termination_reason = "converged"
            break

        #         if np.abs(model1_sent1_prob-step) < 1:
        #             exclusive_write_line(fname,sent1+'.',max_lines=max_sentences)
        #             print('target log-prob achieved, sentence added to file.')
        #             break

        # determine which word are we trying to replace at this step.
        wordi = int(wordis[step])

        if wordi == 0:  # use capitalized words for the first word.
            vocab = vocab_cap
        else:
            vocab = vocab_low

        # for a given word placement, change one sentence at a time (using a random order).
        sentence_indecis_to_modify = list(sentences_to_change).copy()
        random.shuffle(sentence_indecis_to_modify)
        found_replacement_for_at_least_one_sentence = False
        for i_sentence in sentence_indecis_to_modify:
            sentence = sentences[i_sentence]
            words = sentence.split(" ")  # break the sentences into words
            cur_word = words[wordi]  # current word to replace

            def prepare_word_prob_df(models, words, wordi):
                # get word-based log-probs for potential replacement of words[wordi], from each model.
                # some of the models return exact sentence-log-probability for each potential replacement word,
                # others return an approximation.

                all_models_word_df = None
                for i_model, model in enumerate(models):

                    # get a list of potential replacement words. For each word, we have the corresponding sentence log probability.
                    word_list, model_word_probs = get_word_prob_from_model(
                        model, words, wordi
                    )

                    model_words_df = pd.DataFrame(index=word_list)
                    if model.is_word_prob_exact:
                        model_words_df["exact_" + str(i_model)] = model_word_probs
                        model_words_df["approximate_" + str(i_model)] = np.nan
                    else:
                        model_words_df["approximate_" + str(i_model)] = model_word_probs
                        model_words_df["exact_" + str(i_model)] = np.nan

                    # make sure the exact probability for the current word is included (we already have it from the previous round)
                    model_words_df.at[
                        cur_word, "exact_" + str(i_model)
                    ] = sentences_log_p[i_model, i_sentence]

                    # accumulate replacement words and associated probabilities across models
                    if all_models_word_df is None:
                        all_models_word_df = model_words_df
                    else:
                        all_models_word_df = all_models_word_df.join(
                            model_words_df, how="inner"
                        )  # this keeps only the intersecion of the word_lists!
                        # (if we have neither approximate nor exact log-prob of a word for one of the models, we don't consider it).
                return all_models_word_df

            all_models_word_df = prepare_word_prob_df(models, words, wordi)

            if (
                keep_words_unique
            ):  # filter out replacements that would lead to repeating word
                other_words_in_sentence = set(
                    [w.lower() for w in (words[:wordi] + words[wordi + 1 :])]
                )
                if allowed_repeating_words is not None:
                    other_words_in_sentence = other_words_in_sentence - set(
                        allowed_repeating_words
                    )
                candidate_word_is_already_in_sentence = (
                    all_models_word_df.index.str.lower().isin(other_words_in_sentence)
                )
                all_models_word_df = all_models_word_df[
                    ~candidate_word_is_already_in_sentence
                ]

            models_with_approximate_probs = [
                i for i, model in enumerate(models) if not model.is_word_prob_exact
            ]

            # For models with approximate log probabilities, we need
            # at least two datapoints to fit a linear regression from
            # approximate log probs to exact log probs.
            #
            # We'll evaluate the exact probabilties for the words with the maximal
            # and minimal approximate probabilities for these models.
            for i_model in models_with_approximate_probs:
                words_to_evaluate = [
                    all_models_word_df["approximate_" + str(i_model)].idxmax(),
                    all_models_word_df["approximate_" + str(i_model)].idxmin(),
                ]
                for word_to_evaluate in words_to_evaluate:
                    if not np.isnan(
                        all_models_word_df.at[word_to_evaluate, "exact_" + str(i_model)]
                    ):
                        continue  # don't waste time evaluating the word if we already have its exact log prob.
                        # (this might happen if the max probability word is also the current word).
                    modified_words = words.copy()
                    modified_words[wordi] = word_to_evaluate
                    modified_sent = " ".join(modified_words)
                    modified_sent_prob = models[i_model].sent_prob(modified_sent)

                    all_models_word_df.at[
                        word_to_evaluate, "exact_" + str(i_model)
                    ] = modified_sent_prob
                    if verbose >= 4:
                        print(
                            "sentence {}: evaluated {:<30} for model {}".format(
                                i_sentence,
                                cur_word + "→ " + word_to_evaluate,
                                models[i_model].name,
                            )
                        )

            word_list = list(all_models_word_df.index)

            # define loss function for updating sentence_i, with the other sentences fixed.
            loss_func_i = get_loss_fun_with_respect_to_sentence_i(
                sentences_log_p, i_sentence, loss_func
            )

            opt = SetInterpolationSearchPandas(
                loss_fun=loss_func_i, df=all_models_word_df, h_method="LinearRegression"
            )

            # search for the best replacement word

            # first, we'll see if any of words for which we already have exact sentence probabilities improves the loss
            (
                candidate_word_idx,
                candidate_exact_loss,
                new_sent_log_p,
            ) = opt.get_observed_loss_minimum()
            found_useful_replacement = (
                candidate_word_idx is not None
            ) and candidate_exact_loss < current_loss
            if found_useful_replacement:
                candidate_replacement_word = word_list[candidate_word_idx]

            if not found_useful_replacement:
                # none of the replacement words with observed exact sentence log-probabilities improve the current loss.
                # we'll use the approximate sentence log-probabilities to predict a replacement word that will.
                # we'll then observe the exact sentence log-probabilities for that word.
                best_approximate_loss = np.inf
                n_non_decreasing_loss_attempts_per_word = 0

                for word_iteration in range(max_replacement_attempts_per_word):
                    # find the best word according to approximate sentence probabilities
                    (
                        candidate_word_idx,
                        candidate_approximate_loss,
                        missing_models,
                    ) = opt.get_unobserved_loss_minimum()
                    if candidate_word_idx is None:
                        if verbose >= 4:
                            print(
                                "sentence {}: word replacements exhausted.".format(
                                    i_sentence
                                )
                            )
                        break
                    candidate_replacement_word = word_list[candidate_word_idx]

                    # for the best word, we'd like to know the exact sentence log probailities for all of the models
                    modified_words = words.copy()
                    modified_words[wordi] = candidate_replacement_word
                    modified_sent = " ".join(modified_words)
                    for i_model in missing_models:
                        modified_sent_prob = models[i_model].sent_prob(modified_sent)
                        if verbose >= 4:
                            print(
                                "sentence {}: evaluated {:<30} for model {}".format(
                                    i_sentence,
                                    cur_word + "→ " + candidate_replacement_word,
                                    models[i_model].name,
                                )
                            )
                        # update the optimizer
                        opt.update_query_result(
                            xs=[candidate_word_idx], ys=[modified_sent_prob], k=i_model
                        )

                    # get loss for this word using exact estimates
                    candidate_exact_loss, new_sent_log_p = opt.get_loss_for_x(
                        candidate_word_idx
                    )

                    if verbose >= 3:
                        print(
                            "sentence {}: considered {:<30} | loss (current→ approximate/exact): {:.3E}→ {:.3E}/{:.3E}".format(
                                i_sentence,
                                cur_word + "→ " + candidate_replacement_word,
                                current_loss,
                                candidate_approximate_loss,
                                candidate_exact_loss,
                            )
                        )

                    if candidate_exact_loss < current_loss:
                        found_useful_replacement = True
                        break
                    else:
                        if candidate_exact_loss < best_approximate_loss:
                            best_approximate_loss = candidate_exact_loss
                            n_non_decreasing_loss_attempts_per_word = 0
                        else:
                            n_non_decreasing_loss_attempts_per_word += 1
                            if (
                                n_non_decreasing_loss_attempts_per_word
                                > max_non_decreasing_loss_attempts_per_word
                            ):
                                if verbose >= 4:
                                    print(
                                        "sentence {}: {} word replacement failures.".format(
                                            i_sentence,
                                            n_non_decreasing_loss_attempts_per_word,
                                        )
                                    )
                                break

            if found_useful_replacement:
                if verbose >= 2:
                    print(
                        "sentence {}:   replaced {:<30} | loss: {:.3E}→ {:.3E}".format(
                            i_sentence,
                            cur_word + "→ " + candidate_replacement_word,
                            current_loss,
                            candidate_exact_loss,
                        )
                    )

                # update the sentence
                words[wordi] = candidate_replacement_word
                sentence = " ".join(words)
                sentences[i_sentence] = sentence
                current_loss = candidate_exact_loss

                # update the sentence log probabilities
                sentences_log_p[:, i_sentence] = new_sent_log_p
                found_replacement_for_at_least_one_sentence = True

                display_sentences_with_highlighted_word(sentences, i_sentence, wordi)
                if monitoring_func is not None:
                    monitoring_func(sentences, sentences_log_p)
            else:
                if verbose >= 2:
                    print(
                        "sentence {}: no useful replacement for ".format(i_sentence)
                        + cur_word
                        + " (a total of {} possible sentences considered.)".format(
                            len(opt.fully_observed_obs())
                        )
                    )

        # done changing words in all sentences
        if found_replacement_for_at_least_one_sentence:
            n_consequetive_failed_replacements = 0

        #             # a SLOW sanity check of our sentence log probability tracking:
        #             sentences_log_p_=sentences_log_p=get_sentence_log_probabilities(models,sentences)
        #             assert np.isclose(sentences_log_p,sentences_log_p_,rtol=1e-3).all(), 'sanity check failed.'
        else:
            n_consequetive_failed_replacements += 1

    # loop completed (or terminated), organize results.
    if termination_reason == "":
        termination_reason = "max_steps"
    results = {
        "sentences": sentences,
        "sentences_log_p": sentences_log_p,
        "loss": current_loss,
        "step": step,
        "termination_reason": termination_reason,
    }
    return results
