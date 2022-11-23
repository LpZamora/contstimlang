import os
import math
import itertools
import random
import pickle
import re

import pandas as pd
import numpy as np
import torch


from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    BertTokenizerFast,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    RobertaForMaskedLM,
    RobertaTokenizer,
    RobertaTokenizerFast,
    XLMTokenizer,
    XLMWithLMHeadModel,
    ElectraTokenizer,
    ElectraForMaskedLM,
    ElectraTokenizerFast,
)

from knlm import KneserNey

from recurrent_NNs import RNNLM, RNNLM_bilstm, RNNModel

logsoftmax = torch.nn.LogSoftmax(dim=-1)

###############################################################

from vocabulary import vocab_low, vocab_cap

###############################################################


def get_word2id_dict():
    with open(
        os.path.join("model_checkpoints", "neuralnet_word2id_dict.pkl"),
        "rb",
    ) as file:
        word2id = pickle.load(file)

    nn_vocab_size = np.max([word2id[w] for w in word2id]) + 1

    word2id["[MASK]"] = nn_vocab_size
    id2word = dict(zip([word2id[w] for w in word2id], [w for w in word2id]))
    return word2id, nn_vocab_size, id2word


# word2id, nn_vocab_size, id2word = get_word2id_dict()
########################################################


class model_factory:
    """Factory class for creating models"""

    def __init__(self, name, gpu_id, only_tokenizer=False):
        """Initialize the model

        args:
            name: name of the model
            gpu_id: integer id of the gpu to use (or None for cpu)
        """

        self.name = name
        if gpu_id is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{gpu_id}")

        if name == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
            if not only_tokenizer:
                self.model = BertForMaskedLM.from_pretrained("bert-large-cased").to(
                    self.device
                )
            self.is_word_prob_exact = False
        elif name == "bert_new_implementation":
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")
            if not only_tokenizer:
                self.model = BertForMaskedLM.from_pretrained("bert-large-cased").to(
                    self.device
                )
            self.is_word_prob_exact = False
        elif name == "bert_has_a_mouth":
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
            if not only_tokenizer:
                self.model = BertForMaskedLM.from_pretrained("bert-large-cased").to(
                    self.device
                )
            self.is_word_prob_exact = False

        elif name == "bert_whole_word":
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
            if not only_tokenizer:
                self.model = BertForMaskedLM.from_pretrained(
                    "bert-large-cased-whole-word-masking"
                ).to(self.device)
            self.is_word_prob_exact = False

        elif name == "bert_whole_word_has_a_mouth":
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
            if not only_tokenizer:
                self.model = BertForMaskedLM.from_pretrained(
                    "bert-large-cased-whole-word-masking"
                ).to(self.device)
            self.is_word_prob_exact = False

        elif name == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
            if not only_tokenizer:
                self.model = RobertaForMaskedLM.from_pretrained("roberta-large").to(
                    self.device
                )
            self.is_word_prob_exact = False

        elif name == "roberta_new_implementation":
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
            if not only_tokenizer:
                self.model = RobertaForMaskedLM.from_pretrained("roberta-large").to(
                    self.device
                )
            self.is_word_prob_exact = False
        elif name == "roberta_has_a_mouth":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
            if not only_tokenizer:
                self.model = RobertaForMaskedLM.from_pretrained("roberta-large").to(
                    self.device
                )
            self.is_word_prob_exact = False

        elif name == "xlm":
            self.tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-en-2048")
            if not only_tokenizer:
                self.model = XLMWithLMHeadModel.from_pretrained("xlm-mlm-en-2048").to(
                    self.device
                )
            self.is_word_prob_exact = False

        elif name == "electra":
            self.tokenizer = ElectraTokenizer.from_pretrained(
                "google/electra-large-generator"
            )
            if not only_tokenizer:
                self.model = ElectraForMaskedLM.from_pretrained(
                    "google/electra-large-generator"
                ).to(self.device)
            self.is_word_prob_exact = False

        elif name == "electra_new_implementation":
            self.tokenizer = ElectraTokenizerFast.from_pretrained(
                "google/electra-large-generator"
            )
            if not only_tokenizer:
                self.model = ElectraForMaskedLM.from_pretrained(
                    "google/electra-large-generator"
                ).to(self.device)
            self.is_word_prob_exact = False
        elif name == "electra_has_a_mouth":
            self.tokenizer = ElectraTokenizer.from_pretrained(
                "google/electra-large-generator"
            )
            if not only_tokenizer:
                self.model = ElectraForMaskedLM.from_pretrained(
                    "google/electra-large-generator"
                ).to(self.device)
            self.is_word_prob_exact = False

        elif name == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
            if not only_tokenizer:
                self.model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(self.device)
            self.is_word_prob_exact = False

        elif name == "naive_gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
            if not only_tokenizer:
                self.model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(self.device)
            self.is_word_prob_exact = False
        elif name == "plain_gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
            if not only_tokenizer:
                self.model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(self.device)
            self.is_word_prob_exact = False
        elif name == "bilstm":
            self.model = RNNLM_bilstm(
                vocab_size=94608, embed_size=256, hidden_size=256, num_layers=1
            )
            self.model.load_state_dict(
                torch.load(os.path.join("model_checkpoints", "bilstm_state_dict.pt"))
            )
            self.model = self.model.to(self.device)
            word2id, nn_vocab_size, id2word = get_word2id_dict()
            self.word2id = word2id
            self.id2word = id2word
            self.embed_size = 256
            self.hidden_size = 256
            self.vocab_size = nn_vocab_size
            self.num_layers = 1
            self.is_word_prob_exact = False

        elif name == "lstm":
            self.model = RNNLM(
                vocab_size=94607, embed_size=256, hidden_size=512, num_layers=1
            )
            self.model.load_state_dict(
                torch.load(os.path.join("model_checkpoints", "lstm_state_dict.pt"))
            )
            self.model = self.model.to(self.device)
            word2id, nn_vocab_size, id2word = get_word2id_dict()
            self.word2id = word2id
            self.id2word = id2word
            self.embed_size = 256
            self.hidden_size = 512
            self.vocab_size = nn_vocab_size
            self.num_layers = 1
            self.is_word_prob_exact = False

        elif name == "rnn":
            self.model = RNNModel(
                vocab_size=94607, embed_size=256, hidden_size=512, num_layers=1
            )
            self.model.load_state_dict(
                torch.load(os.path.join("model_checkpoints", "rnn_state_dict.pt"))
            )
            self.model = self.model.to(self.device)
            word2id, nn_vocab_size, id2word = get_word2id_dict()
            self.word2id = word2id
            self.id2word = id2word
            self.embed_size = 256
            self.hidden_size = 512
            self.vocab_size = nn_vocab_size
            self.num_layers = 1
            self.is_word_prob_exact = True

        elif name == "trigram":
            self.model = KneserNey.load(
                os.path.join("model_checkpoints", "trigram.model")
            )
            self.is_word_prob_exact = True

        elif name == "bigram":
            self.model = KneserNey.load(
                os.path.join("model_checkpoints", "bigram.model")
            )
            self.is_word_prob_exact = True
        else:
            raise ValueError(f"Model {name} not found")

        if not only_tokenizer:    
            self = get_starts_suffs(self)
            self = get_token_info(self)

    def count_tokens(self, sent):
        if type(sent) in [list, tuple, pd.Series]:
            return [self.count_tokens(s) for s in sent]
        else:
            len_toks=len(self.tokenizer.tokenize(sent))
            return len_toks
            
    def sent_prob(self, sent):

        if self.name in [
            "bert_has_a_mouth",
            "roberta_has_a_mouth",
            "electra_has_a_mouth",
        ]:
            prob = has_a_mouth_sent_prob(self, sent)

        elif self.name in ["bert", "bert_whole_word", "roberta", "xlm", "electra"]:
            prob = bidirectional_transformer_sent_prob(self, sent)

        elif self.name in [
            "bert_new_implementation",
            "roberta_new_implementation",
            "electra_new_implementation",
        ]:
            prob = bidirectional_transformer_sent_prob_new_implementation(self, sent)
        elif self.name == "gpt2":
            prob = gpt2_sent_prob(self, sent)
        elif self.name == "naive_gpt2":
            prob = naive_gpt2_sent_prob(self, sent)
        elif self.name == "plain_gpt2":
            prob = gpt2_sent_scoring_plain(self, sent)
        elif self.name == "bilstm":
            prob = bilstm_sent_prob(self, sent)
        elif self.name == "lstm":
            prob = lstm_sent_prob(self, sent)
        elif self.name == "rnn":
            prob = rnn_sent_prob(self, sent)
        elif self.name == "trigram":
            prob = trigram_sent_prob(self, sent)
        elif self.name == "bigram":
            prob = bigram_sent_prob(self, sent)
        else:
            raise ValueError
        if type(prob) is np.ndarray:
            prob = prob.item()  # return a scalar!

        return prob

    def word_probs(self, words, wordi):

        if self.name in [
            "bert",
            "bert_whole_word",
            "bert_has_a_mouth",
            "bert_new_implementation",
            "roberta",
            "electra",
            "electra_has_a_mouth",
            "electra_new_implementation",
            "roberta_has_a_mouth",
            "roberta_new_implementation",
        ]:
            probs = bidirectional_transformer_word_probs(self, words, wordi)

        elif self.name == "xlm":
            probs = xlm_word_probs(self, words, wordi)

        elif self.name == "gpt2":
            probs = gpt2_word_probs(self, words, wordi)

        elif self.name == "naive_gpt2":
            probs = naive_gpt2_word_probs(self, words, wordi)

        elif self.name == "plain_gpt2":
            probs = naive_gpt2_word_probs(self, words, wordi)

        elif self.name == "bilstm":
            probs = bilstm_word_probs(self, words, wordi)

        elif self.name == "lstm":
            probs = lstm_word_probs(self, words, wordi)

        elif self.name == "rnn":
            probs = rnn_word_probs(self, words, wordi)

        elif self.name == "trigram":
            probs = trigram_word_probs(self, words, wordi)

        elif self.name == "bigram":
            probs = bigram_word_probs(self, words, wordi)
        else:
            raise ValueError
        return probs


def get_starts_suffs(self):

    name = self.name

    if self.name in [
        "bilstm",
        "lstm",
        "rnn",
        "bigram",
        "trigram",
        "plain_gpt2",
    ]:
        return self

    starts = []
    suffs = []

    tokenizer = self.tokenizer

    if name in [
        "bert",
        "bert_has_a_mouth",
        "bert_whole_word",
        "electra",
        "electra_has_a_mouth",
    ]:
        for i in range(len(tokenizer.get_vocab())):
            tok = tokenizer.decode(i)
            if tok[0] != "#":
                starts.append(i)
            elif tok[0] != " ":
                suffs.append(i)
    elif name in ["bert_new_implementation", "electra_new_implementation"]:
        for i in range(tokenizer.vocab_size):
            tok = tokenizer.decode([i])
            if tok[0] != "#":
                starts.append(i)
            elif tok[0] != " ":
                suffs.append(i)
    elif name in [
        "gpt2",
        "roberta",
        "roberta_has_a_mouth",
    ]:
        for i in range(len(tokenizer.get_vocab())):
            tok = tokenizer.decode(i)
            if tok[0] == " " or tok[0] == ".":
                starts.append(i)
            elif tok[0] != " ":
                suffs.append(i)
    elif name == "roberta_new_implementation":
        for i in range(tokenizer.vocab_size):
            tok = tokenizer.decode([i])
            if tok[0] == " " or tok[0] == ".":
                starts.append(i)
            elif tok[0] != " ":
                suffs.append(i)
    elif name in ["xlm"]:
        for i in range(len(tokenizer.get_vocab())):
            tok = tokenizer.convert_ids_to_tokens(i)
            if tok[-4:] == "</w>" and tok != ".</w>":
                suffs.append(i)
            else:
                starts.append(i)
    else:
        raise ValueError
    self.starts = starts
    self.suffs = suffs

    return self


def get_token_info(self):

    name = self.name

    if self.name in ["bilstm", "lstm", "rnn", "bigram", "trigram"]:
        return self

    tokenizer = self.tokenizer
    model = self.model

    if name in ["gpt2", "naive_gpt2", "plain_gpt2"]:

        special_tokens_dict = {"pad_token": "[PAD]"}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        toklist_low = []
        toklist_cap = []

        for v in vocab_low:
            toks = tokenizer.encode(" " + v)
            toklist_low.append(toks)

        for v in vocab_cap:
            toks = tokenizer.encode(" " + v)
            toklist_cap.append(toks)

        self.tokenizer = tokenizer
        self.model = model
        self.toklist_low = toklist_low
        self.toklist_cap = toklist_cap

    else:

        toklist_low = []
        for vi, v in enumerate(vocab_low):
            toks = tokenizer.encode(v)[1:-1]
            toklist_low.append(toks)

        toklist_cap = []
        for vi, v in enumerate(vocab_cap):
            toks = tokenizer.encode(v)[1:-1]
            toklist_cap.append(toks)

        tokparts_all_low = []
        tps_low = []
        for ti, tokens in enumerate(toklist_low):

            tokparts_all = []
            tokens1 = [tokenizer.mask_token_id] * len(tokens)
            tok_perms = list(
                itertools.permutations(np.arange(len(tokens)), len(tokens))
            )

            for perms in tok_perms:

                tokparts = [tokens1]
                tokpart = [tokenizer.mask_token_id] * len(tokens)
                tps_low.append(tokpart.copy())

                for perm in perms[:-1]:

                    tokpart[perm] = tokens[perm]
                    tokparts.append(tokpart.copy())
                    tps_low.append(tokpart.copy())

                tokparts_all.append(tokparts)

            tokparts_all_low.append(tokparts_all)

        tokparts_all_cap = []
        tps_cap = []
        for ti, tokens in enumerate(toklist_cap):

            tokparts_all = []
            tokens1 = [tokenizer.mask_token_id] * len(tokens)
            tok_perms = list(
                itertools.permutations(np.arange(len(tokens)), len(tokens))
            )

            for perms in tok_perms:

                tokparts = [tokens1]
                tokpart = [tokenizer.mask_token_id] * len(tokens)
                tps_cap.append(tokpart.copy())

                for perm in perms[:-1]:

                    tokpart[perm] = tokens[perm]
                    tokparts.append(tokpart.copy())
                    tps_cap.append(tokpart.copy())

                tokparts_all.append(tokparts)

            tokparts_all_cap.append(tokparts_all)

        ######################################################################

        batchsize = 100

        unique_tokparts_low = [list(x) for x in set(tuple(x) for x in tps_low)]

        tokparts_inds_low = []

        vocab_probs_sheet_low = []

        vocab_to_tokparts_inds_low = []

        vocab_to_tokparts_inds_map_low = [
            [] for i in range(int(np.ceil(len(unique_tokparts_low) / batchsize)))
        ]

        for vocind, tokparts_all in enumerate(tokparts_all_low):

            inds_low_all = []
            voc_low_all = []

            toks = toklist_low[vocind]

            tok_perms = list(itertools.permutations(np.arange(len(toks)), len(toks)))

            for ti_all, tokparts in enumerate(tokparts_all):

                tok_perm = tok_perms[ti_all]

                inds_low = []
                voc_low = []
                vocab_to_inds_low = []

                for ti, tokpart in enumerate(tokparts):

                    tokind = tok_perm[ti]

                    ind = unique_tokparts_low.index(tokpart)

                    inds_low.append([ind, tokind, toks[tokind]])

                    voc_low.append(0)

                    vocab_to_inds_low.append([ind, ti_all, ti])

                    batchnum = int(np.floor(ind / batchsize))

                    unique_ind_batch = ind % batchsize

                    vocab_to_tokparts_inds_map_low[batchnum].append(
                        [[vocind, ti_all, ti], [unique_ind_batch, tokind, toks[tokind]]]
                    )

                inds_low_all.append(inds_low)
                voc_low_all.append(voc_low)

            tokparts_inds_low.append(inds_low_all)
            vocab_probs_sheet_low.append(voc_low_all)

            vocab_to_tokparts_inds_low.append(vocab_to_inds_low)

        ######################################################################

        unique_tokparts_cap = [list(x) for x in set(tuple(x) for x in tps_cap)]

        tokparts_inds_cap = []

        vocab_probs_sheet_cap = []

        vocab_to_tokparts_inds_cap = []

        vocab_to_tokparts_inds_map_cap = [
            [] for i in range(int(np.ceil(len(unique_tokparts_cap) / batchsize)))
        ]

        for vocind, tokparts_all in enumerate(tokparts_all_cap):

            inds_cap_all = []
            voc_cap_all = []
            voc_to_inds_cap_all = []

            toks = toklist_cap[vocind]

            tok_perms = list(itertools.permutations(np.arange(len(toks)), len(toks)))

            for ti_all, tokparts in enumerate(tokparts_all):

                tok_perm = tok_perms[ti_all]

                inds_cap = []
                voc_cap = []
                vocab_to_inds_cap = []

                for ti, tokpart in enumerate(tokparts):

                    tokind = tok_perm[ti]

                    ind = unique_tokparts_cap.index(tokpart)

                    inds_cap.append([ind, tokind, toks[tokind]])

                    voc_cap.append(0)

                    vocab_to_inds_cap.append([ind, ti_all, ti])

                    batchnum = int(np.floor(ind / batchsize))

                    unique_ind_batch = ind % batchsize

                    vocab_to_tokparts_inds_map_cap[batchnum].append(
                        [[vocind, ti_all, ti], [unique_ind_batch, tokind, toks[tokind]]]
                    )

                inds_cap_all.append(inds_cap)
                voc_cap_all.append(voc_cap)

            tokparts_inds_cap.append(inds_cap_all)
            vocab_probs_sheet_cap.append(voc_cap_all)

            vocab_to_tokparts_inds_cap.append(vocab_to_inds_cap)

            self.vocab_low = vocab_low
            self.unique_tokparts_low = unique_tokparts_low
            self.vocab_probs_sheet_low = vocab_probs_sheet_low
            self.vocab_to_tokparts_inds_map_low = vocab_to_tokparts_inds_map_low

            self.vocab_cap = vocab_cap
            self.unique_tokparts_cap = unique_tokparts_cap
            self.vocab_probs_sheet_cap = vocab_probs_sheet_cap
            self.vocab_to_tokparts_inds_map_cap = vocab_to_tokparts_inds_map_cap

        return self


def has_a_mouth_sent_prob(self, sent):

    tokenizer = self.tokenizer
    model = self.model

    tokens = tokenizer.tokenize(sent + ".")

    encoded_og = tokenizer.encode(tokens)

    prob = 0

    for i in range(1, len(encoded_og) - 2):

        encoded = encoded_og.copy()

        encoded[i] = tokenizer.mask_token_id

        encoded_cuda = torch.tensor([encoded]).to(self.device)

        with torch.no_grad():
            output = model(input_ids=encoded_cuda)

        probs = output[0][:, i, :][0]

        soft = logsoftmax(probs)

        arr = np.array(soft.cpu().detach())

        prob += arr[encoded_og[i]]

    return prob


def bidirectional_transformer_sent_prob_new_implementation(self, sent):

    if not sent[-1] in [".", ",", "?", "!"]:
        sent = sent.rstrip(" ") + "."

    assert self.tokenizer.is_fast
    result = self.tokenizer.encode_plus(
        sent,
        return_tensors="pt",
        return_token_type_ids=False,
        return_attention_mask=False,
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
    )
    token_ids = result["input_ids"].to(self.device)
    is_special_token = result["special_tokens_mask"].numpy()
    offset_mapping = result["offset_mapping"].numpy()
    assert len(offset_mapping) == 1, "Only batch size 1 is supported"
    offset_mapping = offset_mapping[0]

    def get_word_limits(s):
        # find first and last character of each word
        # e.g. "hello world" -> [(0,5), (6,11)]
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", s)]

    def map_token_ids_to_word_idx(sent, offsets_mapping):
        word_limits = get_word_limits(sent.rstrip(" .,?!"))
        token_id_to_word_idx = [None] * len(offsets_mapping)
        for word_idx, (word_start, word_end) in enumerate(word_limits):
            for token_id_idx, (token_start, token_end) in enumerate(offsets_mapping):
                if (
                    token_start >= word_start
                    and token_end <= word_end
                    and token_end - token_start > 0
                ):
                    token_id_to_word_idx[token_id_idx] = word_idx
        return token_id_to_word_idx

    soft = torch.nn.LogSoftmax(dim=-1)

    def chain_rule_log_prob(token_ids, model, chain_order):
        """evaluate the log probability of a sentence according to a given condition chain order

        args:
        token_ids (list): list of tokens in the sentence
        model (transformer model): the model to use for evaluation
        chain_order (list): list of token indices to condition on in order
            the indexing includes special tokens.
            (e.g. left to right evaluation would be 1,2,3,...,n-2)
            indecis not included are not masked

        returns (float): log probability of the sentence
        """

        if isinstance(token_ids, list):
            n_tokens = len(token_ids)
            token_ids = torch.tensor([token_ids]).to(self.device)
        elif isinstance(token_ids, torch.Tensor):
            n_tokens = token_ids.shape[1]
            token_ids = token_ids.to(self.device)
        else:
            raise ValueError("token_ids must be a list or torch.Tensor")
        assert token_ids.shape == (1, n_tokens)

        n_products = len(chain_order)

        mask_id = self.tokenizer.mask_token_id

        # build the input matrix
        input = token_ids.repeat(n_products, 1)

        # mask the tokens according to the chain order
        for chain_idx, masked_tok_idx in enumerate(chain_order):
            input[: (chain_idx + 1), masked_tok_idx] = mask_id

        # evaluate the model
        with torch.no_grad():
            output = model(input_ids=input)[0]  # (n_products, tokens, vocab)

        # for each chain order, get gather the relevant token (the one we're evaluating)
        output = output[
            torch.arange(n_products), chain_order, :
        ]  # (chain orders, vocab)

        # get the log probabilities of the tokens
        masked_token_id = token_ids[0, chain_order]
        log_probs = soft(output)[
            torch.arange(n_products), masked_token_id
        ]  # (n_products)

        log_likelihood = log_probs.sum()
        return log_likelihood

    # evaluate the sentence in mulitple random orders

    def prepare_word_aware_chain_order(
        token_id_to_word_idx, max_n_chains=100, random_seed=1234
    ):
        """prepare a random chain order for evaluation.

        randomly samples a conditional chain of words.
        with each word, randomly samples a conditional chain of tokens.

        args:
        token_id_to_word_idx (list): a list assinging each token id to a word. special tokens should have None values.
        max_n_chains (int): maximum number of chain orders to evaluate.

        returns (list): list of chain orders

        chain orders are omit special tokens, but are indexed with respect to the complete tokenization"""

        uq_word_indices = sorted(list(set(token_id_to_word_idx) - set([None])))
        n_words = len(uq_word_indices)

        rng = np.random.RandomState(random_seed)

        if math.factorial(n_words) <= max_n_chains:
            word_chain_orders = []
            for word_chain_order in itertools.permutations(uq_word_indices):
                word_chain_orders.append(np.asarray(word_chain_order))
        else:
            word_chain_orders = []
            for _ in range(max_n_chains):
                word_chain_order = rng.permutation(n_words)
                word_chain_orders.append(word_chain_order)

        # transform word-level chains to token-level chains
        token_chain_orders = []
        for word_chain_order in word_chain_orders:
            token_chain_order = []
            for word_idx in word_chain_order:
                cur_word_token_indices = np.flatnonzero(
                    np.array(token_id_to_word_idx) == word_idx
                )
                cur_word_n_tokens = len(cur_word_token_indices)
                if cur_word_n_tokens == 1:
                    token_chain_order.append(cur_word_token_indices[0])
                else:
                    # sample a random permutation of token orders
                    token_permutation = rng.permutation(cur_word_n_tokens)
                    token_chain_order.extend(cur_word_token_indices[token_permutation])
            token_chain_orders.append(token_chain_order)

            # transform indices to account for special tokens
        return token_chain_orders

    token_id_to_word_idx = map_token_ids_to_word_idx(sent, offset_mapping)
    chain_orders = prepare_word_aware_chain_order(
        token_id_to_word_idx, max_n_chains=100
    )

    # alternatively, left-to-right evaluation
    # chain_orders = [np.nonzero(~np.array(is_special_token))[0]]

    all_log_probs = []
    for chain_order in chain_orders:
        all_log_probs.append(chain_rule_log_prob(token_ids, self.model, chain_order))
    average_log_likeihood = torch.stack(all_log_probs).mean().item()
    return average_log_likeihood


def bidirectional_transformer_sent_prob(self, sent):

    tokenizer = self.tokenizer
    model = self.model

    starts = self.starts
    suffs = self.suffs

    word_tokens_per = tokenizer.encode(sent + ".")
    word_tokens_per[-2] = tokenizer.mask_token_id
    in1 = torch.tensor(word_tokens_per).to(self.device).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=in1)[0]
        out = out[:, -2, :]
        out[:, suffs] = math.inf * -1
        soft = logsoftmax(out).cpu().data.numpy()
    per_cent = soft[0, tokenizer.encode(".")[1:-1]]

    words = sent.split(" ")

    word_tokens = tokenizer.encode(sent)[1:-1]

    tokens = tokenizer.encode(sent + ".", add_special_tokens=True)

    start_inds = np.where(np.in1d(tokens, starts) == True)[0][:-2]
    suff_inds = np.where(np.in1d(tokens, suffs) == True)[0]

    wordtoks = [tokenizer.encode(w)[1:-1] for w in words]

    tokens_all = []
    labels_all = []

    input_to_mask_inds = dict()

    word_inds = list(np.linspace(1, len(words), len(words)).astype("int"))

    msk_inds_all = []

    for i in range(1, len(words) + 1):
        msk_inds = list(itertools.combinations(word_inds, i))
        msk_inds = [list(m) for m in msk_inds]
        msk_inds_all = msk_inds_all + msk_inds

    msk_inds_all = msk_inds_all[::-1]

    for mski, msk_inds in enumerate(msk_inds_all):

        msk_inds = list(np.array(msk_inds) - 1)

        msk_inds_str = "".join([str(m) for m in msk_inds])

        tokens1 = [[]]
        labels1 = []

        for j in range(len(words)):

            if j in msk_inds:

                wordtok = wordtoks[j]
                tokens1c = tokens1.copy()

                msk_inds_str1 = msk_inds_str + "_" + str(j)

                tokens1 = [
                    tokens + [tokenizer.mask_token_id] * len(wordtok)
                    for tokens in tokens1
                ]

                tok_orders = [
                    list(itertools.combinations(np.arange(len(wordtok)), x))
                    for x in range(1, len(wordtok))
                ]
                tok_orders = [list(item) for sublist in tok_orders for item in sublist]

                tokens2 = []

                for tok_order in tok_orders:
                    for tokens in tokens1c[:1]:
                        for toki, tok in enumerate(wordtok):

                            if toki in tok_order:
                                tokens = tokens + [tok]
                            else:
                                tokens = tokens + [tokenizer.mask_token_id]

                        tokens2.append(tokens)

                tokens1 = tokens1 + tokens2

                if len(wordtok) > 1:

                    perms = list(
                        itertools.permutations(np.arange(len(wordtok)), len(wordtok))
                    )

                    input_to_mask_inds[msk_inds_str1] = []

                    for perm in perms:

                        temprows = []

                        perm = list(perm)

                        for pi in range(len(perm)):

                            perm1 = perm[:pi]
                            perm1sort = list(np.sort(perm1))

                            if len(perm1sort) == 0:

                                row1 = len(tokens_all)
                                row2 = len(tokens1c[0]) + perm[pi]

                            else:

                                row1_offset = tok_orders.index(perm1sort) + 1
                                row1 = len(tokens_all) + row1_offset
                                row2 = len(tokens1c[0]) + perm[pi]

                            row3 = row2
                            rows = [row1, row2, row3]
                            temprows.append(rows)

                        input_to_mask_inds[msk_inds_str1].append(temprows)

                else:

                    row1 = len(tokens_all)
                    row2 = len(tokens1c[0])
                    row3 = row2

                    rows = [row1, row2, row3]

                    input_to_mask_inds[msk_inds_str1] = [[rows]]

            else:

                tokens1 = [tokens + wordtoks[j] for tokens in tokens1]

        tokens_all = tokens_all + tokens1

    tokens_all = [
        [tokenizer.cls_token_id]
        + t
        + [tokenizer.encode(".")[1:-1][0], tokenizer.sep_token_id]
        for t in tokens_all
    ]

    inputs = torch.tensor(tokens_all).to(self.device)

    batchsize = 100

    with torch.no_grad():

        if len(inputs) < batchsize:

            out = model(input_ids=inputs)[0]

            out = out[:, 1:-2, :]

            for x in range(out.shape[1]):
                if x in start_inds[1:]:
                    out[:, x - 1, suffs] = math.inf * -1
                elif x in suff_inds[1:]:
                    out[:, x - 1, starts] = math.inf * -1

            soft = logsoftmax(out)

            soft = soft[:, :, word_tokens]

        else:

            for b in range(int(np.ceil(len(inputs) / batchsize))):
                in1 = inputs[batchsize * b : batchsize * (b + 1)]
                lab1 = labels_all[batchsize * b : batchsize * (b + 1)]
                out1 = model(input_ids=in1)[0]

                out1 = out1[:, 1:-2, :]

                for x in range(out1.shape[1]):
                    if x in start_inds[1:]:
                        out1[:, x - 1, suffs] = math.inf * -1
                    elif x in suff_inds[1:]:
                        out1[:, x - 1, starts] = math.inf * -1

                soft1 = logsoftmax(out1)

                soft1 = soft1[:, :, word_tokens]

                if b == 0:
                    soft = soft1

                else:
                    soft = torch.cat((soft, soft1))

                try:
                    torch.cuda.empty_cache()
                except:
                    pass

        orders = list(itertools.permutations(word_inds, i))

        orders = random.Random(1234).sample(orders, min(len(orders), 100))

        for orderi, order in enumerate(orders):

            for ordi, ind in enumerate(order):

                curr_masked = np.sort(order[ordi:])

                key = (
                    "".join([str(c - 1) for c in curr_masked]) + "_" + str(ind - 1)
                )  # -1 to correct for CLS

                out_inds_all = input_to_mask_inds[key]

                for oi_all, out_inds in enumerate(out_inds_all):

                    for oi, out_ind in enumerate(out_inds):

                        prob = soft[out_ind[0], out_ind[1], out_ind[2]]

                        if oi == 0:
                            word_probs = prob.unsqueeze(0)
                        else:
                            word_probs = torch.cat((word_probs, prob.unsqueeze(0)), 0)

                    word_probs_prod = torch.sum(word_probs)

                    if oi_all == 0:
                        word_probs_all = word_probs_prod.unsqueeze(0)
                    else:
                        word_probs_all = torch.cat(
                            (word_probs_all, word_probs_prod.unsqueeze(0)), 0
                        )

                word_prob = torch.mean(word_probs_all)

                if ordi == 0:
                    chain_prob = word_prob.unsqueeze(0)
                else:
                    chain_prob = torch.cat((chain_prob, word_prob.unsqueeze(0)), 0)

            chain_prob_prod = torch.sum(chain_prob)

            assert chain_prob_prod != 0

            if orderi == 0:
                chain_probs = chain_prob_prod.unsqueeze(0)
            else:
                chain_probs = torch.cat((chain_probs, chain_prob_prod.unsqueeze(0)), 0)

        score = np.mean(chain_probs.cpu().data.numpy()) + per_cent

        return score


def bidirectional_transformer_word_probs(self, words, wordi):

    tokenizer = self.tokenizer
    model = self.model

    name = self.name
    starts = self.starts
    suffs = self.suffs

    if wordi > 0:
        vocab = self.vocab_low
        unique_tokparts = self.unique_tokparts_low
        vocab_probs_sheet = self.vocab_probs_sheet_low
        vocab_to_tokparts_inds_map = self.vocab_to_tokparts_inds_map_low
    else:
        vocab = self.vocab_cap
        unique_tokparts = self.unique_tokparts_cap
        vocab_probs_sheet = self.vocab_probs_sheet_cap
        vocab_to_tokparts_inds_map = self.vocab_to_tokparts_inds_map_cap

    words = words.copy()

    words[wordi] = tokenizer.mask_token

    sent = " ".join(words)

    tokens = tokenizer.encode(sent + ".")

    mask_ind = tokens.index(tokenizer.mask_token_id)

    tok1 = tokens[:mask_ind]
    tok2 = tokens[mask_ind + 1 :]

    inputs = []
    for un in unique_tokparts:

        in1 = tok1 + un + tok2
        inputs.append(in1)

    maxlen = np.max([len(i) for i in inputs])

    inputs = [i + [0] * (maxlen - len(i)) for i in inputs]

    att_mask = [[1] * len(i) + [0] * (maxlen - len(i)) for i in inputs]

    inputs = torch.tensor(inputs).to(self.device)
    att_mask = torch.tensor(att_mask, dtype=torch.float32).to(self.device)

    batchsize = 100

    for i in range(int(np.ceil(len(inputs) / batchsize))):

        vocab_to_tokparts_inds_map_batch = vocab_to_tokparts_inds_map[i]

        inputs1 = inputs[batchsize * i : batchsize * (i + 1)]

        att_mask1 = att_mask[batchsize * i : batchsize * (i + 1)]

        with torch.no_grad():

            out1 = model(inputs1, attention_mask=att_mask1)[0]

            out1 = out1[:, mask_ind : mask_ind + 6, :]

            out1[:, 0, suffs] = math.inf * -1
            out1[:, 1:, starts] = math.inf * -1

            soft = logsoftmax(out1)

            for vti in vocab_to_tokparts_inds_map_batch:

                vocab_probs_sheet[vti[0][0]][vti[0][1]][vti[0][2]] = float(
                    soft[vti[1][0], vti[1][1], vti[1][2]]
                )

            del soft

    vocab_probs = []
    for x in range(len(vocab_probs_sheet)):

        probs = []
        for y in range(len(vocab_probs_sheet[x])):

            prob = np.sum(vocab_probs_sheet[x][y])

            probs.append(prob)

        vocab_probs.append(np.mean(probs))

    vocab_probs = np.array(vocab_probs)

    return vocab_probs


def xlm_word_probs(self, words, wordi):

    tokenizer = self.tokenizer
    model = self.model

    name = self.name
    starts = self.starts
    suffs = self.suffs

    if wordi > 0:
        unique_tokparts = self.unique_tokparts_low
        vocab_probs_sheet = self.vocab_probs_sheet_low.copy()
        vocab_to_tokparts_inds_map = self.vocab_to_tokparts_inds_map_low
    else:
        unique_tokparts = self.unique_tokparts_cap
        vocab_probs_sheet = self.vocab_probs_sheet_cap.copy()
        vocab_to_tokparts_inds_map = self.vocab_to_tokparts_inds_map_cap

    words = words.copy()  # Don't change the input argument!

    words[wordi] = tokenizer.mask_token

    sent = " ".join(words)

    tokens = tokenizer.encode(sent + ".")

    mask_ind = tokens.index(tokenizer.mask_token_id)

    tok1 = tokens[:mask_ind]
    tok2 = tokens[mask_ind + 1 :]

    inputs = []
    for un in unique_tokparts:

        in1 = tok1 + un + tok2
        inputs.append(in1)

    maxlen = np.max([len(i) for i in inputs])

    att0s_all = [maxlen - len(i) for i in inputs]

    inputs = [[0] * (maxlen - len(i)) + i for i in inputs]

    att_mask = [[0] * (maxlen - len(i)) + [1] * len(i) for i in inputs]

    inputs = torch.tensor(inputs).to(self.device)
    att_mask = torch.tensor(att_mask, dtype=torch.float32).to(self.device)

    batchsize = 100

    for i in range(int(np.ceil(len(inputs) / batchsize))):

        vocab_to_tokparts_inds_map_batch = vocab_to_tokparts_inds_map[i]

        inputs1 = inputs[batchsize * i : batchsize * (i + 1)]

        att0s = att0s_all[batchsize * i : batchsize * (i + 1)]

        att_mask1 = att_mask[batchsize * i : batchsize * (i + 1)]

        with torch.no_grad():

            out1 = model(inputs1, attention_mask=att_mask1)[0]

            out1[:, -1 * (len(tokens) - mask_ind), starts] = math.inf * -1
            out1[:, : -1 * (len(tokens) - mask_ind) - 1, suffs] = math.inf * -1

            out2 = torch.zeros([batchsize, 6, out1.shape[2]])

            for x in range(len(inputs1)):

                out2[
                    x,
                    : out1[x, mask_ind + att0s[x] : mask_ind + 6 + att0s[x], :].shape[
                        0
                    ],
                    :,
                ] = out1[x, mask_ind + att0s[x] : mask_ind + 6 + att0s[x], :]

            soft = logsoftmax(out2)

            for vti in vocab_to_tokparts_inds_map_batch:

                vocab_probs_sheet[vti[0][0]][vti[0][1]][vti[0][2]] = float(
                    soft[vti[1][0], vti[1][1], vti[1][2]]
                )

            del soft

    vocab_probs = []
    for x in range(len(vocab_probs_sheet)):

        probs = []
        for y in range(len(vocab_probs_sheet[x])):

            prob = np.sum(vocab_probs_sheet[x][y])

            probs.append(prob)

        vocab_probs.append(np.mean(probs))

    vocab_probs = np.array(vocab_probs)

    return vocab_probs


def gpt2_sent_prob(self, sent):

    tokenizer = self.tokenizer
    model = self.model

    starts = self.starts
    suffs = self.suffs

    sent = ". " + sent + "."

    tokens = tokenizer.encode(sent)
    inputs = torch.tensor(tokens).to(self.device)

    with torch.no_grad():
        out = model(inputs)

    unsoft = out[0]
    lab1 = inputs.cpu().data.numpy()

    probs = []
    for x in range(len(lab1) - 1):

        lab = lab1[x + 1]
        unsoft1 = unsoft[x]

        if lab in starts:

            soft = logsoftmax(unsoft1[starts])
            prob = float(soft[starts.index(lab)].cpu().data.numpy())

        elif lab in suffs:

            soft = logsoftmax(unsoft1[suffs])
            prob = float(soft[suffs.index(lab)].cpu().data.numpy())

        probs.append(prob)

    prob = np.sum(probs)

    return prob


def gpt2_word_probs(self, words, wordi):

    tokenizer = self.tokenizer
    model = self.model

    starts = self.starts
    suffs = self.suffs

    if wordi == 0:
        vocab = vocab_cap
        toklist = self.toklist_cap
    else:
        vocab = vocab_low
        toklist = self.toklist_low

    sent1 = " ".join(words[:wordi])
    sent2 = " ".join(words[wordi + 1 :])

    tok1 = tokenizer.encode(". " + sent1)
    tok2 = tokenizer.encode(" " + sent2)

    ####################################################3##

    lp = 0
    while 0 == 0:
        in1 = tok1
        in1 = torch.tensor(in1).to(self.device)

        with torch.no_grad():
            out1 = model(in1)[0]
            soft1 = torch.softmax(out1, -1)[-1].cpu().data.numpy()

        logsoft1 = np.log(soft1)

        tops = np.where(logsoft1 > -10 - lp * 5)[0]

        tops = [t for t in tops if t in starts]

        if len(tops) < 10:
            lp = lp + 1
        else:
            break

    ##########################

    inputs = []
    vocab_tops = []
    vocab_tops_ind = []

    for wi, word in enumerate(vocab):

        wordtok = toklist[wi]

        if wordtok[0] in tops:

            vocab_tops.append(word)
            vocab_tops_ind.append(wi)

            in1 = tok1 + wordtok + tok2 + tokenizer.encode(".")

            inputs.append(in1)

    maxlen = np.max([len(i) for i in inputs])

    inputs0 = [i + [0] * (maxlen - len(i)) for i in inputs]
    att_mask = np.ceil(np.array(inputs0) / 100000)

    inputs = [i + [tokenizer.pad_token_id] * (maxlen - len(i)) for i in inputs]

    batchsize = 64

    for i in range(int(np.ceil(len(inputs) / batchsize))):

        inputs1 = np.array(inputs[batchsize * i : batchsize * (i + 1)])

        att_mask1 = att_mask[batchsize * i : batchsize * (i + 1)]

        inputs2 = torch.tensor(inputs1).to(self.device)
        att_mask1 = torch.tensor(att_mask1, dtype=torch.float32).to(self.device)

        with torch.no_grad():

            out1 = model(input_ids=inputs2, attention_mask=att_mask1)[0]

            out_suff_inds = torch.where(
                torch.tensor(np.in1d(inputs1, suffs).reshape(inputs1.shape[0], -1)).to(
                    self.device
                )
                == True
            )

            out_start_inds = torch.where(
                torch.tensor(np.in1d(inputs1, starts).reshape(inputs1.shape[0], -1)).to(
                    self.device
                )
                == True
            )

            for x in range(len(out_suff_inds[0])):
                out1[out_suff_inds[0][x], out_suff_inds[1][x] - 1, starts] = (
                    math.inf * -1
                )

            for x in range(len(out_start_inds[0])):
                out1[out_start_inds[0][x], out_start_inds[1][x] - 1, suffs] = (
                    math.inf * -1
                )

            soft = logsoftmax(out1)

            for v in range(len(inputs1)):

                numwords = len(np.where(inputs1[v] < tokenizer.pad_token_id)[0]) - 1

                probs = torch.tensor(
                    [soft[v, n, inputs1[v][n + 1]] for n in range(0, numwords)]
                )

                prob = torch.sum(probs)

                if i == 0 and v == 0:
                    vocab_probs = prob.unsqueeze(0)
                else:
                    vocab_probs = torch.cat((vocab_probs, prob.unsqueeze(0)), 0)

    vocab_probs = vocab_probs.cpu().data.numpy()

    return vocab_probs, vocab_tops_ind


def naive_gpt2_sent_prob(self, sent):

    tokenizer = self.tokenizer
    model = self.model

    #     starts=self.starts
    #     suffs=self.suffs

    sent = ". " + sent
    if sent[-1] != ".":
        sent = sent + "."

    tokens = tokenizer.encode(sent)
    inputs = torch.tensor(tokens).to(self.device)

    with torch.no_grad():
        out = model(inputs)

    unsoft = out[0]
    lab1 = inputs.cpu().data.numpy()

    probs = []
    for x in range(len(lab1) - 1):

        lab = lab1[x + 1]
        unsoft1 = unsoft[x]

        soft = logsoftmax(unsoft1)
        prob = float(soft[lab].cpu().data.numpy())
        probs.append(prob)

    prob = np.sum(probs)

    return prob


def naive_gpt2_word_probs(self, words, wordi):

    tokenizer = self.tokenizer
    model = self.model

    # starts=self.starts
    # suffs=self.suffs

    if wordi == 0:
        vocab = vocab_cap
        toklist = self.toklist_cap
    else:
        vocab = vocab_low
        toklist = self.toklist_low

    sent1 = " ".join(words[:wordi])
    sent2 = " ".join(words[wordi + 1 :])

    tok1 = tokenizer.encode(". " + sent1)
    tok2 = tokenizer.encode(" " + sent2)

    ####################################################3##

    lp = 0
    while 0 == 0:
        in1 = tok1
        in1 = torch.tensor(in1).to(self.device)

        with torch.no_grad():
            out1 = model(in1)[0]
            soft1 = torch.softmax(out1, -1)[-1].cpu().data.numpy()

        logsoft1 = np.log(soft1)

        tops = np.where(logsoft1 > -10 - lp * 5)[0]

        # tops=[t for t in tops if t in starts]

        if len(tops) < 10:
            lp = lp + 1
        else:
            break

    ##########################

    inputs = []
    vocab_to_input_inds = []
    vocab_to_input_pred_vocs = []
    vocab_to_input_pos = []

    vocab_tops = []
    vocab_tops_ind = []

    for wi, word in enumerate(vocab):

        wordtok = toklist[wi]

        if wordtok[0] in tops:

            vocab_tops.append(word)
            vocab_tops_ind.append(wi)

            in1 = tok1 + wordtok + tok2 + tokenizer.encode(".")

            inputs.append(in1)

    maxlen = np.max([len(i) for i in inputs])

    inputs0 = [i + [0] * (maxlen - len(i)) for i in inputs]
    att_mask = np.ceil(np.array(inputs0) / 100000)

    inputs = [i + [tokenizer.pad_token_id] * (maxlen - len(i)) for i in inputs]

    batchsize = 64

    for i in range(int(np.ceil(len(inputs) / batchsize))):

        inputs1 = np.array(inputs[batchsize * i : batchsize * (i + 1)])

        att_mask1 = att_mask[batchsize * i : batchsize * (i + 1)]

        inputs2 = torch.tensor(inputs1).to(self.device)
        att_mask1 = torch.tensor(att_mask1, dtype=torch.float32).to(self.device)

        with torch.no_grad():

            out1 = model(input_ids=inputs2, attention_mask=att_mask1)[0]
            soft = logsoftmax(out1)

            for v in range(len(inputs1)):

                numwords = len(np.where(inputs1[v] < tokenizer.pad_token_id)[0]) - 1

                # probs=torch.tensor([soft[v,n,inputs1[v][n+1]] for n in range(len(tok1)-1,numwords)])

                probs = torch.tensor(
                    [soft[v, n, inputs1[v][n + 1]] for n in range(0, numwords)]
                )

                prob = torch.sum(probs)  # .cpu().data.numpy())

                if i == 0 and v == 0:
                    vocab_probs = prob.unsqueeze(0)
                else:
                    vocab_probs = torch.cat((vocab_probs, prob.unsqueeze(0)), 0)

    vocab_probs = vocab_probs.cpu().data.numpy()

    return vocab_probs, vocab_tops_ind


def gpt2_sent_scoring_plain(self, lines, batch_size=32):

    if type(lines) == str:
        return gpt2_sent_scoring_plain(self, [lines], batch_size=batch_size)[0]

    lines = [". " + l for l in lines]
    lines = [l if l[-1] == "." else l + "." for l in lines]
    tokenizer = self.tokenizer
    model = self.model

    if len(lines) > batch_size:

        def chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i : i + n]

        chunks = list(chunks(lines, batch_size))
        scores, n_tokens = [], []
        for chunk in chunks:
            scores_, n_tokens_ = gpt2_sent_scoring_plain(
                tokenizer, model, chunk, batch_size
            )
            scores.extend(scores_)
            n_tokens.extend(n_tokens_)
        return scores

    # lines = [tokenizer.eos_token + line for line in lines]
    tok_res = tokenizer.batch_encode_plus(lines, return_tensors="pt", padding=True)
    input_ids = tok_res["input_ids"]
    attention_mask = tok_res["attention_mask"]
    with torch.no_grad():
        lines_len = torch.sum(tok_res["attention_mask"], dim=1)

        outputs = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            labels=input_ids.to(model.device),
        )
        loss, logits = outputs[:2]
        scores, n_tokens = [], []
        for line_ind in range(len(lines)):
            line_log_prob = 0.0
            for token_ind in range(lines_len[line_ind] - 1):
                token_prob = torch.nn.functional.softmax(
                    logits[line_ind, token_ind], dim=0
                )
                token_id = input_ids[line_ind, token_ind + 1]
                line_log_prob += torch.log(token_prob[token_id])
            scores.append(line_log_prob.item())
            n_tokens.append(lines_len[line_ind].item() - 1)
    return scores


def bilstm_word_probs(self, words, wordi):

    model = self.model

    hidden_size = self.hidden_size
    embed_size = self.embed_size
    vocab_size = self.vocab_size
    num_layers = self.num_layers

    if wordi > 0:
        vocab = vocab_low
    else:
        vocab = vocab_cap

    states = (
        torch.zeros(2, 1, hidden_size).to(self.device),
        torch.zeros(2, 1, hidden_size).to(self.device),
    )

    ids = [self.word2id[w] for w in words] + [self.word2id["."]]

    ids[wordi] = self.word2id["[MASK]"]

    ids = torch.tensor(ids).to(self.device).unsqueeze(0)

    out, states = model(ids, states, 0, [wordi])

    soft = logsoftmax(out[0]).cpu().data.numpy()

    soft = soft[[self.word2id[v] for v in vocab]]

    return soft


def bilstm_sent_prob(self, sent):

    model = self.model

    hidden_size = self.hidden_size
    embed_size = self.embed_size
    vocab_size = self.vocab_size
    num_layers = self.num_layers

    words = sent.split()

    word_ids = [self.word2id[w] for w in words]

    tok_orders = [
        list(itertools.combinations(np.arange(len(words)), x))
        for x in range(1, len(words))
    ]
    tok_orders = [""] + [item for sublist in tok_orders for item in sublist]

    chains = []
    order_to_chain = dict()

    for i, tok_order in enumerate(tok_orders):

        base = [self.word2id["[MASK]"]] * len(words) + [self.word2id["."]]

        for t in tok_order:
            base[t] = word_ids[t]

        chains.append(base)

        key = "".join([str(t) for t in tok_order])

        order_to_chain[key] = i

    chains = torch.tensor(chains).to(self.device)

    states = (
        torch.zeros(2, chains.shape[0], hidden_size).to(self.device),
        torch.zeros(2, chains.shape[0], hidden_size).to(self.device),
    )

    out, states = model(chains, states, 0, np.arange(chains.shape[0] * chains.shape[1]))

    soft = logsoftmax(out)

    soft = soft[:, word_ids]

    soft = soft.reshape(chains.shape[0], chains.shape[1], soft.shape[1])

    tok_perms = list(itertools.permutations(np.arange(len(words))))

    tok_perms100 = random.Random(1234).sample(tok_perms, min(500, len(tok_perms)))

    probs_all = []

    for tok_perm in tok_perms100:

        probs = []

        for tpi, tp in enumerate(tok_perm):

            key = "".join([str(t) for t in np.sort(tok_perm[:tpi])])

            key_ind = order_to_chain[key]

            chain = chains[key_ind]

            prob = float(torch.sum(soft[key_ind, tp, tp]))

            probs.append(prob)

        probs_all.append(np.sum(probs))

    prob = np.mean(probs_all)

    return prob


def lstm_word_probs(self, words, wordi):

    model = self.model

    hidden_size = self.hidden_size
    embed_size = self.embed_size
    vocab_size = self.vocab_size
    num_layers = self.num_layers

    if wordi > 0:
        vocab = vocab_low
    else:
        vocab = vocab_cap

    wordi = wordi + 1

    words = ["."] + words + ["."]

    states = (
        torch.zeros(num_layers, 1, hidden_size).to(self.device),
        torch.zeros(num_layers, 1, hidden_size).to(self.device),
    )

    inputs = torch.tensor([self.word2id[w] for w in words]).to(self.device).unsqueeze(0)
    outputs, states = model(inputs, states, 0)
    soft = logsoftmax(outputs).cpu().data.numpy()

    ss = np.argsort(soft[wordi - 1])[::-1]
    top_words = [self.id2word[s] for s in ss]
    top_words = list(set(top_words) & set(vocab))
    inds = [vocab.index(t) for t in top_words]

    probs = []

    for wi, w in enumerate(top_words):

        states = (
            torch.zeros(num_layers, 1, hidden_size).to(self.device),
            torch.zeros(num_layers, 1, hidden_size).to(self.device),
        )

        words[wordi] = w

        prob = lstm_sent_prob(self, " ".join(words[1:-1]))
        probs.append(prob)

    probs = np.array(probs)

    return probs, inds


def lstm_sent_prob(self, sent):

    model = self.model

    hidden_size = self.hidden_size
    embed_size = self.embed_size
    vocab_size = self.vocab_size
    num_layers = self.num_layers

    states = (
        torch.zeros(num_layers, 1, hidden_size).to(self.device),
        torch.zeros(num_layers, 1, hidden_size).to(self.device),
    )

    words = ["."] + sent.split() + ["."]

    inputs = torch.tensor([self.word2id[w] for w in words]).to(self.device).unsqueeze(0)

    outputs, states = model(inputs, states, 0)

    soft = logsoftmax(outputs).cpu().data.numpy()

    prob = np.sum([float(soft[wi, self.word2id[w]]) for wi, w in enumerate(words[1:])])

    return prob


def rnn_sent_prob(self, sent):

    model = self.model

    hidden_size = self.hidden_size
    embed_size = self.embed_size
    vocab_size = self.vocab_size
    num_layers = self.num_layers

    states = (
        torch.zeros(num_layers, 1, hidden_size).to(self.device),
        torch.zeros(num_layers, 1, hidden_size).to(self.device),
    )

    words = ["."] + sent.split() + ["."]

    inputs = torch.tensor([self.word2id[w] for w in words]).to(self.device).unsqueeze(0)

    h0 = torch.zeros(num_layers, 1, hidden_size).to(self.device)

    outputs, states = model(inputs, h0)

    soft = logsoftmax(outputs).cpu().data.numpy()[0]

    prob = np.sum([float(soft[wi, self.word2id[w]]) for wi, w in enumerate(words[1:])])

    return prob


def rnn_word_probs(self, words, wordi):

    model = self.model

    hidden_size = self.hidden_size
    embed_size = self.embed_size
    vocab_size = self.vocab_size
    num_layers = self.num_layers

    if wordi > 0:
        vocab = vocab_low
    else:
        vocab = vocab_cap

    wordi = wordi + 1

    words = ["."] + words + ["."]

    h0 = torch.zeros(num_layers, 1, hidden_size).to(self.device)

    inputs = torch.tensor([self.word2id[w] for w in words]).to(self.device).unsqueeze(0)
    outputs, states = model(inputs, h0)
    soft = logsoftmax(outputs).cpu().data.numpy()[0]

    ss = np.argsort(soft[wordi - 1])[::-1]
    top_words = [self.id2word[s] for s in ss]
    top_words = list(set(top_words) & set(vocab))
    inds = [vocab.index(t) for t in top_words]

    probs = []

    for wi, w in enumerate(top_words):

        states = (
            torch.zeros(num_layers, 1, hidden_size).to(self.device),
            torch.zeros(num_layers, 1, hidden_size).to(self.device),
        )

        words[wordi] = w

        prob = rnn_sent_prob(self, " ".join(words[1:-1]))

        probs.append(prob)

    probs = np.array(probs)

    return probs, inds


def trigram_sent_prob(self, sent):

    words = sent.split()

    model = self.model

    words = ["<BOS1>", "<BOS2>"] + words + [".", "<EOS1>"]

    prob = model.evaluateSent(words)

    return prob


def trigram_word_probs(self, words, wordi):

    model = self.model

    words = ["<BOS1>", "<BOS2>"] + words + [".", "<EOS1>"]

    if wordi == 0:
        vocab = vocab_cap
    else:
        vocab = vocab_low

    probs = []
    for w in vocab:

        words[wordi + 2] = w

        prob = model.evaluateSent(words)

        probs.append(prob)

    probs = np.array(probs)

    return probs


def bigram_sent_prob(self, sent):

    words = sent.split()

    model = self.model

    words = ["<BOS2>"] + words + ["."]

    prob = model.evaluateSent(words)

    return prob


def bigram_word_probs(self, words, wordi):

    model = self.model

    words = ["<BOS2>"] + words + ["."]

    if wordi == 0:
        vocab = vocab_cap
    else:
        vocab = vocab_low

    probs = []
    for w in vocab:

        words[wordi + 1] = w

        prob = model.evaluateSent(words)

        probs.append(prob)

    probs = np.array(probs)

    return probs
