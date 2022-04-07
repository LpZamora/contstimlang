#!/usr/bin/env python
# coding: utf-8

import random

import numpy as np
import pandas as pd

from interpolation_search import SetInterpolationSearchPandas
from vocabulary import vocab_low, vocab_low_freqs, vocab_cap, vocab_cap_freqs


def controversiality_loss_func(sentences_log_p):
    """
    Given a reference natural sentence, form a variant of it which is at least as likely according to one model,
    and as unlikely as possible according to the other.

    this is a penality method implementation of Eq. 4 in the preprint.

    args:
    sentences_log_p (numpy.array,  M_models x S_sentences or D_designs x M_models x S_sentences)
    current log probabilities such that sentences_log_p[m,s]=log_p(sentence_s|model_m)
    # D_designs is for evaluating more than one trial (i.e. a sentence pair) at the same time.
    comment: currently, both M_models and S_sentences must be 2
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


class exhaustive_word_location_dispenser:
    def __init__(self, sent_len):
        self.sent_len = sent_len
        self.on_successful_replacement()

    def on_successful_replacement(self):
        self.not_attempted_locations = list(range(self.sent_len))
        if hasattr(self, "last_returned"):
            self.not_attempted_locations.remove(self.last_returned)
        random.shuffle(self.not_attempted_locations)

    def get_next_word_location(self):
        """return a word location for replacement (int)"""
        if len(self.not_attempted_locations) > 0:
            self.last_returned = int(self.not_attempted_locations.pop(0))
            return self.last_returned
        else:
            return None


class cyclic_word_location_dispenser:
    def __init__(self, sent_len, max_steps):
        self.sent_len = sent_len
        self.locations_list = []
        while len(self.locations_list) <= max_steps:
            wordi = np.arange(sent_len)
            random.shuffle(wordi)
            self.locations_list = self.locations_list + list(wordi)
        self.on_successful_replacement()

    def on_successful_replacement(self):
        self.attempts_counter = 0

    def get_next_word_location(self):
        """return a word location for replacement (int)"""
        self.attempts_counter += 1
        if self.attempts_counter > self.sent_len:
            return None
        elif len(self.locations_list) == 0:
            return None
        else:
            return int(self.locations_list.pop(0))


def update_history(
    history, sentences, sentences_log_p=None, loss=None, model_names=None
):
    """update the history of sentence optimization
    args:
        history (pd.DataFrame) history of sentence optimization
        sentences (list of strings)
        sentences_log_p (numpy.array, optional) sentence_log_p[m,s] is the log-probabilitiy assigned to sentence s by model m.
        loss (float), optional
        model_names (list of strings, optional) used for column headers for sentence probabilities

    return updated history
    """

    n_sentences = len(sentences)
    new_row = pd.DataFrame()

    for i_sentence, sentence in enumerate(sentences):
        new_row[f"s{i_sentence}"] = [sentence]
    if sentences_log_p is not None:
        n_models = sentences_log_p.shape[0]
        assert n_sentences == sentences_log_p.shape[1]
        if model_names is not None:
            assert n_models == len(model_names)
        for i_model in range(n_models):
            for i_sentence in range(n_sentences):
                if model_names is not None:
                    model_name = model_names[i_model]
                else:
                    model_name = f"m{i_model}"
                new_row[f"p_s{i_sentence}_{model_name}"] = [
                    sentences_log_p[i_model, i_sentence]
                ]
    if loss is not None:
        new_row["loss"] = [float(loss)]

    history = pd.concat([history, new_row], axis=0)
    return history


def optimize_sentence_set(
    n_sentences,
    models,
    loss_func,
    sent_len=8,
    sentences_to_change=None,
    replacement_strategy="cyclic",
    sentences=None,
    initial_sampling="uniform",
    start_with_identical_sentences=True,
    max_steps=10000,
    internal_stopping_condition=lambda loss: False,
    external_stopping_check=lambda: False,
    max_replacement_attempts_per_word=50,
    max_non_decreasing_loss_attempts_per_word=5,
    keep_words_unique=False,
    allowed_repeating_words=None,
    monitoring_func=None,
    save_history=False,
    model_names=None,
    verbose=3,
):
    """Optimize a sentence set of n_sentences.
        n_sentences (int) how many sentences to optimize (e.g., 2 for a sentence pair).
        models (list) a list of model objects
        loss_func (function) return a loss to be minimized given a [d,ms] log_p(sentence s|model m, d d)
        sentence_len (int) how many words
        sentences_to_change (list) which sentences should be optimized (e.g., [0,1] for the first two sentences). Default: all sentences.
        replacement_strategy (str) how to choose the word to replace. 'exhaustive' (don't stop optimizing until all word replacements failed)
            or - 'cyclic' (follow cyclic word orders, as described in the preprint)
    # initialization:
        sentences (list) initial sentences (if None, sentences are randomly sampled).
        initial_sampling (str) 'uniform' or 'proportional' (when sentences is None, how to sample words)
        start_with_identical_sentences (bool) if True, start with identical sentences.
    # top-level stopping conditions:
        max_steps (int) maximum total number of word replacement attempts
        internal_stopping_condition (function) function for internal stopping condition. args: loss - finish optimization if True
        external_stopping_check (function) if this function returns True, abort the optimization (no args) - used to check if other process finished the same optimization task
    # word replacement-level stopping conditions:
        max_replacement_attempts_per_word (int) if not None, stop trying to replace a word after max_replacement_attempts_per_word alternative sentences were evaluated
        max_non_decreasing_loss_attempts_per_word (int) if not None, stop trying to replace a word after max_non_decreasing_loss_attempts_per_word sentences did not show decreasing loss
    # non-model based constraints:
        keep_words_unique (bool) all words must be unique (within sentence)
        allowed_repeating_words (list) if keep_words_unique is True, these words are excluded from the uniqueness constrain. The words should be lower-case.
    # monitoring:
        save_history (bool) if True, save the history of the optimization
        model_names (list of strings) used to for history column headers
        monitoring_func (function) function for online user feedback. args: (sentences, sentences_log_p)
        verbose (int)
    """

    if replacement_strategy == "cyclic":
        word_location_dispenser = cyclic_word_location_dispenser(sent_len, max_steps)
    elif replacement_strategy == "exhaustive":
        word_location_dispenser = exhaustive_word_location_dispenser(sent_len)
    else:
        raise ValueError("replacement_strategy must be 'exhaustive' or 'cyclic'")

    if sentences is not None:
        # sentences provided. do some sanity checks.
        n_sentences = len(sentences)
        sentence_len = np.unique([len(sentence.split(" ")) for sentence in sentences])
        assert len(sentence_len) == 1, "all sentences should be of the same length"
        sentence_len = sentence_len.item()
    else:  # initialize random sentences
        if start_with_identical_sentences:
            sentences = [
                initialize_random_word_sentence(
                    sent_len, initial_sampling=initial_sampling
                )
            ] * n_sentences
        else:
            sentences = [
                initialize_random_word_sentence(
                    sent_len, initial_sampling=initial_sampling
                )
                for _ in range(n_sentences)
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

    termination_reason = ""

    if save_history:
        history = pd.DataFrame()
        history = update_history(
            history,
            sentences,
            sentences_log_p=sentences_log_p,
            model_names=model_names,
            loss=current_loss,
        )
    for step in range(max_steps):

        # check stopping conditions
        if external_stopping_check():
            # abort optimization (e.g., another worker completed the sentence).
            return False

        if internal_stopping_condition(current_loss):
            termination_reason = "internal_stopping_condition"
            break

        # determine which word are we trying to replace at this step.
        wordi = word_location_dispenser.get_next_word_location()
        if wordi is None:
            termination_reason = "no more locations"
            break

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

                if save_history:
                    history = update_history(
                        history,
                        sentences,
                        sentences_log_p=sentences_log_p,
                        model_names=model_names,
                        loss=current_loss,
                    )
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
            word_location_dispenser.on_successful_replacement()

        #             # a SLOW sanity check of our sentence log probability tracking:
        #             sentences_log_p_=sentences_log_p=get_sentence_log_probabilities(models,sentences)
        #             assert np.isclose(sentences_log_p,sentences_log_p_,rtol=1e-3).all(), 'sanity check failed.'

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
    if save_history:
        results["history"] = history
    return results
