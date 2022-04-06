#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import random
import math

from lstm_class import RNNLM
from bilstm_class import RNNLM_bilstm
from rnn_class import RNNModel
from model_functions import model_factory

models = [
    "bigram",
    "trigram",
    "rnn",
    "lstm",
    "bilstm",
    "bert",
    "bert_whole_word",
    "roberta",
    "xlm",
    "electra",
    "gpt2",
]

rngs = np.load("prob_ranges.npy")

# model names
model1_name = "trigram"
model2_name = "rnn"

rng1 = rngs[models.index(model1_name)]
rng2 = rngs[models.index(model2_name)]

# bigram and trigram models run on CPU, so gpu_id will be ignored
model1_gpu_id = 0
model2_gpu_id = 0

# squashing thresholds for two models
squash_threshold1 = rng1[0] * -1 - 10
squash_threshold2 = rng2[0] * -1 - 10

# sentence length
sent_len = 8


# In[ ]:


model1 = model_factory(model1_name, model1_gpu_id)
model2 = model_factory(model2_name, model2_gpu_id)

get_model1_sent_prob = model1.sent_prob
get_model2_sent_prob = model2.sent_prob
get_model1_word_probs = model1.word_probs
get_model2_word_probs = model2.word_probs


# In[3]:


from vocabulary import vocab_low, vocab_low_freqs, vocab_cap, vocab_cap_freqs


# In[4]:


def squash(prob, squash_threshold):
    prob = (
        10 * np.log(1 + math.e ** ((prob + squash_threshold) / 10)) - squash_threshold
    )
    return prob


def cont_score(
    model1_sent1_prob, model1_sent2_prob, model2_sent1_prob, model2_sent2_prob
):

    gamma = 100  # subject noise

    model1_sent1_prob = squash(np.log(model1_sent1_prob), squash_threshold1)
    model1_sent2_prob = squash(np.log(model1_sent2_prob), squash_threshold1)
    model2_sent1_prob = squash(np.log(model2_sent1_prob), squash_threshold2)
    model2_sent2_prob = squash(np.log(model2_sent2_prob), squash_threshold2)

    s1a_b = model1_sent1_prob - model1_sent2_prob
    s2a_b = model2_sent1_prob - model2_sent2_prob

    # Prob that model 1 is better and a subject picks a/b
    p1a = (1 / 2) / (1 + np.exp(-(s1a_b) / gamma))
    p1b = 1 / 2 - p1a

    # Prob that model 2 is better and a subject picks a/b
    p2a = (1 / 2) / (1 + np.exp(-(s2a_b) / gamma))
    p2b = 1 / 2 - p2a

    # Mutual information of model and sentence pick
    # Each term is p(model,sent)*log(p(model,sent)/(p(model)*p(sent)))
    conta = p1a * np.log2(p1a / ((p1a + p2a) / 2)) + p2a * np.log2(
        p2a / ((p1a + p2a) / 2)
    )

    contb = p1b * np.log2(p1b / ((p1b + p2b) / 2)) + p2b * np.log2(
        p2b / ((p1b + p2b) / 2)
    )

    return conta, contb


# In[7]:


wordi = np.arange(sent_len)
wordis = []
for i in range(1000):
    random.shuffle(wordi)
    wordis = wordis + list(wordi)

words1 = list(np.random.choice(vocab_cap, 1, p=vocab_cap_freqs)) + list(
    np.random.choice(vocab_low, sent_len - 1, p=vocab_low_freqs, replace=False)
)
words2 = words1.copy()


words1o = words1.copy()
words2o = words2.copy()

sent1 = " ".join(words1)
sent2 = " ".join(words2)


model1_sent1_prob = get_model1_sent_prob(sent1)
model2_sent1_prob = get_model2_sent_prob(sent1)

model1_sent2_prob = get_model1_sent_prob(sent2)
model2_sent2_prob = get_model2_sent_prob(sent2)


conta_last, contb_last = cont_score(
    model1_sent1_prob, model1_sent2_prob, model2_sent1_prob, model2_sent2_prob
)


prob_diff_last12 = np.log(model1_sent1_prob / model2_sent1_prob)
prob_diff_last21 = np.log(model2_sent2_prob / model1_sent2_prob)

probs_all = [model1_sent1_prob, model2_sent1_prob, model1_sent2_prob, model2_sent2_prob]

probs_all_list = []
print(probs_all)
print(sent1)
print(sent2)
print("\n")


for samp in range(10000):

    words1o = words1.copy()
    words2o = words2.copy()

    wordi = int(wordis[samp])

    cur_word1 = words1[wordi]
    cur_word2 = words2[wordi]

    if wordi == 0:
        vocab = vocab_cap
    else:
        vocab = vocab_low

    model1_word1_probs = get_model1_word_probs(words1, wordi)
    model1_word2_probs = get_model1_word_probs(words2, wordi)

    if len(model1_word1_probs) == 2:
        model1_word1_inds = model1_word1_probs[1]
        model1_word1_probs = model1_word1_probs[0]
        model1_word2_inds = model1_word2_probs[1]
        model1_word2_probs = model1_word2_probs[0]
    else:
        model1_word1_inds = np.arange(len(vocab))
        model1_word2_inds = np.arange(len(vocab))

    words1 = words1o.copy()
    words2 = words2o.copy()

    model2_word1_probs = get_model2_word_probs(words1, wordi)
    model2_word2_probs = get_model2_word_probs(words2, wordi)

    if len(model2_word1_probs) == 2:
        model2_word1_inds = model2_word1_probs[1]
        model2_word1_probs = model2_word1_probs[0]
        model2_word2_inds = model2_word2_probs[1]
        model2_word2_probs = model2_word2_probs[0]
    else:
        model2_word1_inds = np.arange(len(vocab))
        model2_word2_inds = np.arange(len(vocab))

    word1_inds = list(set(model1_word1_inds) & set(model2_word1_inds))
    word2_inds = list(set(model1_word2_inds) & set(model2_word2_inds))

    model1_word1_probs = [
        model1_word1_probs[wi]
        for wi, i in enumerate(word1_inds)
        if i in model1_word1_inds
    ]
    model1_word2_probs = [
        model1_word2_probs[wi]
        for wi, i in enumerate(word2_inds)
        if i in model1_word2_inds
    ]
    model2_word1_probs = [
        model2_word1_probs[wi]
        for wi, i in enumerate(word1_inds)
        if i in model2_word1_inds
    ]
    model2_word2_probs = [
        model2_word2_probs[wi]
        for wi, i in enumerate(word2_inds)
        if i in model2_word2_inds
    ]

    word1_list = [vocab[w] for w in word1_inds]
    word2_list = [vocab[w] for w in word2_inds]

    model1_word1_probs = model1_word1_probs / np.sum(model1_word1_probs)
    model1_word2_probs = model1_word2_probs / np.sum(model1_word2_probs)

    model2_word1_probs = model2_word1_probs / np.sum(model2_word1_probs)
    model2_word2_probs = model2_word2_probs / np.sum(model2_word2_probs)

    word1_probs12 = model1_word1_probs * np.log(model1_word1_probs / model2_word1_probs)
    word2_probs21 = model2_word2_probs * np.log(model2_word2_probs / model1_word2_probs)

    word1_probs12[np.isnan(word1_probs12)] = 0
    word1_probs12[np.where(word1_probs12 < 0)[0]] = 0
    word1_probs12 = word1_probs12 / np.sum(word1_probs12)

    word2_probs21[np.isnan(word2_probs21)] = 0
    word2_probs21[np.where(word2_probs21 < 0)[0]] = 0
    word2_probs21 = word2_probs21 / np.sum(word2_probs21)

    word1_tops = [word1_list[vp] for vp in np.argsort(word1_probs12)[::-1][:10]] + [
        cur_word1
    ]
    word2_tops = [word2_list[vp] for vp in np.argsort(word2_probs21)[::-1][:10]] + [
        cur_word2
    ]

    model1_sent1_probs = []
    model1_sent2_probs = []
    model2_sent1_probs = []
    model2_sent2_probs = []

    sent1_conts12 = []
    sent2_conts21 = []

    for word1, word2 in zip(word1_tops, word2_tops):

        words1t = words1o.copy()
        words2t = words2o.copy()

        words1t[wordi] = word1
        words2t[wordi] = word2

        sent1t = " ".join(words1t)
        sent2t = " ".join(words2t)

        model1_sent1t_prob = get_model1_sent_prob(sent1t)
        model2_sent1t_prob = get_model2_sent_prob(sent1t)

        model1_sent2t_prob = get_model1_sent_prob(sent2t)
        model2_sent2t_prob = get_model2_sent_prob(sent2t)

        model1_sent1_probs.append(model1_sent1t_prob)
        model1_sent2_probs.append(model1_sent2t_prob)

        model2_sent1_probs.append(model2_sent1t_prob)
        model2_sent2_probs.append(model2_sent2t_prob)

    conta_scores = []
    contb_scores = []
    cont_score_inds = []
    a = -1
    for m11, m21 in zip(model1_sent1_probs, model2_sent1_probs):
        a += 1
        b = -1
        for m12, m22 in zip(model1_sent2_probs, model2_sent2_probs):
            b += 1

            conta, contb = cont_score(m11, m12, m21, m22)
            conta_scores.append(conta)
            contb_scores.append(contb)
            cont_score_inds.append([a, b])

    aa = cont_score_inds[np.argmax(conta_scores)][0]
    bb = cont_score_inds[np.argmax(contb_scores)][1]

    new_word1 = word1_tops[aa]
    new_word2 = word2_tops[bb]

    new_word1o = new_word1
    new_word2o = new_word2

    words1[wordi] = new_word1.upper()
    words2[wordi] = new_word2.upper()

    sent1p = " ".join(words1)
    sent2p = " ".join(words2)

    words1[wordi] = new_word1.lower()
    if wordi == 0 or new_word1o[0].isupper():
        words1[wordi] = new_word1.lower().capitalize()

    words2[wordi] = new_word2.lower()
    if wordi == 0 or new_word2o[0].isupper():
        words2[wordi] = new_word2.lower().capitalize()

    sent1 = " ".join(words1)
    sent2 = " ".join(words2)

    model1_sent1_prob = get_model1_sent_prob(sent1)
    model2_sent1_prob = get_model2_sent_prob(sent1)

    model1_sent2_prob = get_model1_sent_prob(sent2)
    model2_sent2_prob = get_model2_sent_prob(sent2)

    probs_all = [
        model1_sent1_prob,
        model2_sent1_prob,
        model1_sent2_prob,
        model2_sent2_prob,
    ]

    probs_all_list.append(probs_all)

    print(probs_all)
    print(sent1p)
    print(sent2p)
    print("\n")


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:
