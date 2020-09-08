#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import random
import math
import pathlib

import numpy as np
import portalocker

from lstm_class import RNNLM
from bilstm_class import RNNLM_bilstm
from rnn_class import RNNModel
from model_functions import model_factory
from interpolation_search import SetInterpolationSearch

models=['bigram','trigram','rnn','lstm','bilstm','bert','bert_whole_word','roberta','xlm','electra','gpt2']
#models=['gpt2']

# printing verbosity level
verbose=3

max_sentences=60

#bigram and trigram models run on CPU, so gpu_id will be ignored
model1_gpu_id=0

#sentence length
sent_len=8

def get_n_lines(fname):
    if not os.path.exists(fname):
        return 0
    else:
        with open(fname,'r') as fh:
            return sum(1 for line in fh)

def exclusive_write_line(fname,line,max_lines):
    if not os.path.exists(os.path.dirname(fname)):
        pathlib.Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(fname, mode='a+') as fh:
        n_lines_in_files=sum(1 for line in fh)
        if n_lines_in_files>=max_lines:
            print('max lines ('+str(max_lines) + ') in ' + fname + ' reached, not writing.')
        fh.write(line+'\n')
        fh.flush()
        os.fsync(fh.fileno())

with open('vocab_low.pkl', 'rb') as file:
    vocab_low=pickle.load(file)

with open('vocab_low_freqs.pkl', 'rb') as file:
    vocab_low_freqs=pickle.load(file)

with open('vocab_cap.pkl', 'rb') as file:
    vocab_cap=pickle.load(file)

with open('vocab_cap_freqs.pkl', 'rb') as file:
    vocab_cap_freqs=pickle.load(file)

model_loaded=False
for model1_name in models:
    for step_ind in range(10):

        fname=os.path.join('synthesized_sentences',
            'single_model_sentences_'+str(sent_len)+'_word'
            ,model1_name+'_level_'+str(step_ind+1)+'.txt')

        if get_n_lines(fname)>=max_sentences:
            continue

        if not model_loaded:
            print("loading "+model1_name + " into gpu " + str(model1_gpu_id))
            model1=model_factory(model1_name,model1_gpu_id)

            get_model1_sent_prob=model1.sent_prob
            get_model1_word_probs=model1.word_probs

            #get probability range for model 1
            rng=np.load('prob_ranges_10_90.npz')[model1_name]
            low=rng[0]-65
            high=rng[1]
            steps=np.linspace(low,high,10)

            model_loaded=True

        step=steps[step_ind]

        while get_n_lines(fname)<max_sentences:

            num_sents=get_n_lines(fname)

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

                if get_n_lines(fname)>=max_sentences:
                    break

                if np.abs(model1_sent1_prob-step) < 1:
                    exclusive_write_line(fname,sent1+'.',max_lines=max_sentences)
                    print('target log-prob achieved, sentence added to file.')
                    break

                elif cycle==sent_len:
                    print('premature convergence, restarting')
                    break

                # elif model1_sent1_prob - step > 1:
                #     print('overshoot log-prob, restarting')
                #     break

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

                are_we_going_up=model1_sent1_prob<step

                # setup word-level optimization *******
                if cur_word1 in word1_list:
                    # current word is included in model word-log-prob. Use it to inform the model
                    cur_word_idx=word1_list.index(cur_word1)
                    if are_we_going_up: # if the target log-probability is lower than the current log-probability,
                        highest_logprob_word_idx=np.argmax(model1_word1_probs) # check first the highest log-probability word
                        if highest_logprob_word_idx!=cur_word_idx:
                            initial_xs_guesses=[highest_logprob_word_idx]
                        else: # unless it's the same as the current word, then look for the second best
                            second_highest_logprob_word_idx=np.argpartition(-model1_word1_probs,1)[1]
                            initial_xs_guesses=[second_highest_logprob_word_idx]
                    else: # going down
                        lowest_logprob_word_idx=np.argmin(model1_word1_probs) # check first the lowest log-probability word
                        if lowest_logprob_word_idx!=cur_word_idx:
                            initial_xs_guesses=[lowest_logprob_word_idx]
                        else: # unless it's the same as the current word, then look for the second lowest
                            second_lowest_logprob_word_idx=np.argpartition(model1_word1_probs,1)[1]
                            initial_xs_guesses=[second_lowest_logprob_word_idx]

                    initial_observed_xs=[cur_word_idx]
                    initial_observed_ys=[model1_sent1_prob]

                else: # current word is not included in model word-log-prob. use bounds for an initial estimate
                    highest_logprob_word_idx=np.argmax(model1_word1_probs)
                    lowest_logprob_word_idx=np.argmin(model1_word1_probs) # check first the lowest log-probability word

                    if are_we_going_up:
                        initial_xs_guesses=[highest_logprob_word_idx,lowest_logprob_word_idx]
                    else:
                        initial_xs_guesses=[lowest_logprob_word_idx,highest_logprob_word_idx]

                    initial_observed_xs=[]
                    initial_observed_ys=[]

                loss_fun=lambda log_p: abs(log_p-step)

                opt=SetInterpolationSearch(loss_fun=loss_fun,
                    g=model1_word1_probs, # we approximate sentence log-probabilities by the word log-probabilities
                    initial_observed_xs=initial_observed_xs,initial_observed_ys=initial_observed_ys,
                    initial_xs_guesses=initial_xs_guesses,
                    h_method='LinearRegression')

                cur_loss=loss_fun(model1_sent1_prob)
                n_words_evaluated_without_loss_improvement=0
                max_n_words_to_consider_without_loss_improvement=50
                found_useful_replacement=False
                best_loss_so_far=np.inf

                while (not found_useful_replacement) and (n_words_evaluated_without_loss_improvement<max_n_words_to_consider_without_loss_improvement):
                    next_word_idx=opt.yield_next_x()
                    word1=word1_list[next_word_idx]

                    # evaluate the sentence log-probability with the next word
                    words1t=words1o.copy()
                    words1t[wordi]=word1
                    sent1t=' '.join(words1t)
                    model1_sent1t_prob=get_model1_sent_prob(sent1t)

                    # update the optimizer
                    opt.update_query_result(xs=[next_word_idx],ys=[model1_sent1t_prob])

                    loss_with_replacement=loss_fun(model1_sent1t_prob)

                    if loss_with_replacement<best_loss_so_far:
                        best_loss_so_far=loss_with_replacement
                        n_words_evaluated_without_loss_improvement=0
                    else:
                        n_words_evaluated_without_loss_improvement+=1

                    if loss_with_replacement<cur_loss:
                        found_useful_replacement=True

                    if verbose>=3:
                        print("{:<40} | log-prob: {:06.1f}→{:06.1f} | loss: {:06.1f}→ {:06.1f} | {:02d} words without loss improvement.".format(
                            cur_word1+'→ '+word1,model1_sent1_prob,model1_sent1t_prob,cur_loss,loss_with_replacement,n_words_evaluated_without_loss_improvement))

                # matplotlib plot of sentence probabilities as function of word probabilities
                # if n_words_evaluated_without_loss_improvement>5:
                #     opt.debugging_figure()

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
                        print('no useful replacement for ' + cur_word1)
                    cycle+=1
    if model_loaded:
        del model1
        del get_model1_sent_prob
        del get_model1_word_probs
        del steps
        model_loaded=False