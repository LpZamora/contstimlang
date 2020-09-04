#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import random
import math

from lstm_class import RNNLM
from bilstm_class import RNNLM_bilstm
from rnn_class import RNNModel
from model_functions import model_factory

models=['bigram','trigram','rnn','lstm','bilstm','bert','bert_whole_word','roberta','xlm','electra','gpt2']

#load probability ranges and get steps
rngs=np.load('prob_ranges.npy')
steps_all=np.zeros([11,10])
for i,rng in enumerate(rngs):
    low=rng[0]-45
    high=rng[1]-5
    steps=np.linspace(low,high,10)
    steps_all[i,:]=steps

# printing verbosity level
verbose=3

#model names
model1_name='gpt2'

#bigram and trigram models run on CPU, so gpu_id will be ignored
model1_gpu_id=0

#get probability range for model 1
steps=steps_all[models.index(model1_name)]

#sentence length
sent_len=8

model1=model_factory(model1_name,model1_gpu_id)

get_model1_sent_prob=model1.sent_prob
get_model1_word_probs=model1.word_probs

with open('vocab_low.pkl', 'rb') as file:
    vocab_low=pickle.load(file)

with open('vocab_low_freqs.pkl', 'rb') as file:
    vocab_low_freqs=pickle.load(file)

with open('vocab_cap.pkl', 'rb') as file:
    vocab_cap=pickle.load(file)

with open('vocab_cap_freqs.pkl', 'rb') as file:
    vocab_cap_freqs=pickle.load(file)

for step_ind in range(10):

    step=steps[step_ind]

    with open(model1_name+'_level_'+str(step_ind+1)+'.txt','w') as file: # close the file when script quits/breaks

        num_sents=0
        while num_sents<60:

            wordi=np.arange(sent_len)
            wordis=[]
            for i in range(1000):
                random.shuffle(wordi)
                wordis=wordis+list(wordi)

            vocab_low_freqs1=np.ones([len(vocab_low_freqs)])/len(vocab_low_freqs)
            vocab_cap_freqs1=np.ones([len(vocab_cap_freqs)])/len(vocab_cap_freqs)

            words1=list(np.random.choice(vocab_cap, 1, p=vocab_cap_freqs1)) + list(np.random.choice(vocab_low, sent_len-1, p=vocab_low_freqs1, replace=False))

            words1o=words1.copy()

            sent1=' '.join(words1)

            model1_sent1_prob=get_model1_sent_prob(sent1)

            if verbose>=2:
                print('\n')
                print('initialized sentence '+str(num_sents))
                print('Target: '+str(step))
                print('Current: '+str(model1_sent1_prob))
                print(sent1)

            probs_all_list=[model1_sent1_prob]

            cycle=0

            for samp in range(10000):

                if np.abs(model1_sent1_prob-step) < 1:

                    file.write(sent1+'.')
                    file.write('\n')

                    num_sents+=1
                    print('target log-prob achieved, sentence added to file.')
                    break

                elif cycle==sent_len:
                    print('premature convergence, restarting')
                    break

                elif model1_sent1_prob - step > 1:
                    print('overshoot log-prob, restarting')
                    break

                if samp%sent_len==0:
                    cycle=0

                words1o=words1.copy()

                wordi=int(wordis[samp])

                cur_word1=words1[wordi]

                if wordi==0:
                    vocab=vocab_cap
                else:
                    vocab=vocab_low

                model1_word1_probs = get_model1_word_probs(words1,wordi)

                if len(model1_word1_probs)==2:
                    model1_word1_inds=model1_word1_probs[1]
                    model1_word1_probs=model1_word1_probs[0]
                else:
                    model1_word1_inds=np.arange(len(vocab))

                words1=words1o.copy()

                word1_list=[vocab[w] for w in model1_word1_inds]

    #             model1_word1_probs=model1_word1_probs/np.sum(model1_word1_probs)

                max_n_words_to_consider=50
                word1_tops=[word1_list[vp] for vp in np.argsort(model1_word1_probs)[::-1][:max_n_words_to_consider] if word1_list[vp]!= cur_word1 ] # + [cur_word1] # why do we need to incldue cur_word1 here?

                model1_sent1_probs=[]
                model1_sent1_prob_diffs=[]

                sent1_conts12=[]
                sent2_conts21=[]

                found_useful_replacement=False
                for word1 in word1_tops:

                    words1t=words1o.copy()

                    words1t[wordi]=word1

                    sent1t=' '.join(words1t)

                    model1_sent1t_prob=get_model1_sent_prob(sent1t)

                    model1_sent1_probs.append(model1_sent1t_prob)

                    if (model1_sent1t_prob>model1_sent1_prob) and (model1_sent1t_prob - step <= 1):
                        found_useful_replacement=True
                        break
                    else:
                        if verbose>=3:
                            if model1_sent1t_prob<=model1_sent1_prob:
                                print(cur_word1 + '->' + word1 + ' does not improve log-prob (' + str(model1_sent1t_prob) + '<=' + str(model1_sent1_prob) + ')')
                            else:
                                print(cur_word1 + '->' + word1 + ' overshots log-prob (' + str(model1_sent1t_prob) + '>=' + str(step) + '+1)')

                    #model1_sent1_prob_diffs.append(np.abs(model1_sent1t_prob-step))

                #aa=np.argmin(model1_sent1_prob_diffs)
                #new_word1=word1_tops[aa]
                if found_useful_replacement:
                    new_word1=word1
                    new_word1o=new_word1

                    words1[wordi]=new_word1.upper()

                    sent1p=' '.join(words1)

                    words1[wordi]=new_word1.lower()
                    if wordi==0 or new_word1o[0].isupper():
                        words1[wordi]=new_word1.lower().capitalize()

                    sent1=' '.join(words1)

                    model1_sent1_prob=model1_sent1t_prob

                    if verbose>=2:
                        print('Target: '+str(step))
                        print('Current: '+str(model1_sent1_prob))
                        print(sent1p)
                else:
                    if verbose>=2:
                        print('no useful replacement in top '+str(len(word1_tops))+' words for ' + cur_word1)
                    cycle+=1

                probs_all_list.append(model1_sent1_prob)
