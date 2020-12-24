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

from lstm_class import RNNLM
from bilstm_class import RNNLM_bilstm
from rnn_class import RNNModel
from model_functions import model_factory
from interpolation_search import SetInterpolationSearch

from vocabulary import vocab_low, vocab_low_freqs, vocab_cap, vocab_cap_freqs
    
def external_stopping_checking():
    if get_n_lines(fname)>=max_sentences:
        if verbose>=3:
            print("found "+str(get_n_lines(fname)) + " lines in " + fname)
        return True
    else:
        return False

def log_likelihood_target_stopping_condition(loss):                
    if np.abs(loss) < 1:        
        print('target log-prob achieved, sentence added to file.')
        return True
    else:
        return False
        
        #exclusive_write_line(fname,sent1+'.',max_lines=max_sentences)

def get_word_prob_from_model(model,words,wordi):
    """ evaluate the log-probability of sentences resulting from replacing the word at location wordi in the sentence defined by 'words'.
        words (list of strings)
        wordi (int) (0 for first word and so on).
        
        returns:
        word list (list)
        model_word_probs (np.Array) exact or approximate log-probability of each sentence resulting from replacing the word at wordi with a word from word_list
        prob_type (str) either 'exact' or 'approximate'        
        
        #TODO: move this into the models' word_prob methods.
    """
    
    if wordi==0:
        vocab=vocab_cap
    else:
        vocab=vocab_low
                    
    output = model.word_prob(words,wordi)    
    if len(output)==2: # model returns indecis and probabilities
        model_word_inds=model_word_probs[1]
        model_word_probs=model_word_probs[0]
    else: # probabilities are returned for all vocab
        model_word_inds=np.arange(len(vocab))
        assert len(model_word_probs)==len(vocab) 
                
    word_list=[vocab[w] for w in model1_word1_inds]

    if model1_name in ['gpt2','rnn','lstm']:
        # these models currently return a long list of actual sentence probability for their top-words
        prob_type='exact'
    else: # for other model, word-prob is an approximation of sentence probability
        prob_type='approximate'

    return word_list, model_word_probs, prob_type

def controversiality_score(decision_p,model_prior=None):
    """ return the mutual information between model identities and predicted sentence choices for a single trial.
    
    args:
    decision_p (numpy.array) (D_designs, M_models, S_sentences) tensor such that decision_p[d,m,s] equals to log-p(sentence_s | model_m, design d).
    model_prior (numpy.array) M_models long vector of model prior probabilities
    
    returns:
    score (numpy.array) M_designs-long vector of scores.    
    """
    
    D_designs, M_models, S_sentences=decision_log_p.shape
    
    if model_prior is None:
        model_prior=np.log(np.ones(shape=(M_models))/M_models)
    model_prior=model_prior.reshape((1,M_models,1))
            
    # I(X,Y)=D_KL(P(X,Y) || P(X)*P(Y) )
    
    joint=decision_p*model_prior # joint[d,m,s]=p(sentence_s, model_m | design_d)
    sentence_marginal=joint.sum(axis=1,keepdims=True) # summate the joint over model to produce sentence_marginal[d,0,s]=p(sentence_s | design_d)
    
    marginal_product=sentence_marginal*model_prior # marginal_product[d,m,s]=p(sentence_s | design_d)*p(model_m)
                                
    MI=scipy.stats.entropy(pk=joint.reshape((M_designs,-1)),qk=product.reshape((M_designs,-1)),base=2,axis=1) # MI[d]=D_KL(joint[d],marginal_product[d])
    return MI

def human_choice_controversiality_loss(sentences_log_p,human_choice_response_models=None,model_names=None):
    """ calculate model identification loss given sentence log probabilities and human_choice_response_models        
    
        args:
        sentences_log_p (numpy.array, D_designs x M_models x S_sentences) current log probabilities, such that sentences_log_p[m,s]=log_p(sentence_s|model_m)
                
    """
    
    D_designs, M_models, S_sentences = sentences_log_p.shape
    if human_choice_response_models is None: # direct readout of choice probabilities (i.e., no squashing, gamma = 1.0)
        decision_log_p=sentences_log_p-logsumexp(sentences_log_p,axis=2,keepdims=True) # normalize sentence probability by total sentence probability (within model and design)
        
    if isinstance(human_choice_response_models,ModelToHumanDecision):
        assert S_sentences==2, 'ModelToHumanDecision currently assumes only two sentences'        
        decision_log_p=np.empty_like(sentences_log_p)
        for i_model in range(M_models):            
            decision_log_p[:,i_model,:]=human_choice_response_models.decision_NLL(log_p1=sentences_log_p[:,i_model,0],
                                                      log_p2=sentences_log_p[:,i_model,1],
                                                      model=model_names[i_model])                    
    else isinstance(human_choice_response_models,list):
        # model-specific human response model object
        assert S_sentences==2, 'ModelToHumanDecision currently assumes only two sentences'                
        assert len(human_choice_response_models)==M_models
        
        decision_log_p=np.empty_like(sentences_log_p)
        for i_model in range(M_models):            
            decision_log_p[:,i_model,:]=human_choice_response_models.decision_NLL(log_p1=sentences_log_p[:,i_model,0],
                                                      log_p2=sentences_log_p[:,i_model,1])
        
    decision_p=np.exp(decision_log_p)
        
        
                          

def get_loss_fun_with_respect_to_sentence_i(sentences_log_p,i_sentence,loss_func):
    """ return a loss function with respect to updating the model probabilities of just one of the sentences (the i_sentence-th).
    This help function simplifies loss-evaluation when we seek to update one of the sentences.
    
    args:
    sentences_log_p (numpy.array , M_models x S_sentences) current log probabilities, such that sentences_log_p[m,s]=log_p(sentence_s|model_m)
    i_sentence (int) which sentence are we considering to change
    loss_func (function) a function that maps design-model-sentence log-probabilities tensor (D_designs x M_models x S_sentence) to D_designs-long loss vector    
        
    # returns:
    loss_fun_i (function) a function that maps D_designs-long vector of sentence log-probabilities to D_designs-long loss vector
    """
    
    def loss_fun_i(sentence_i_log_p):
        """ return loss as function of updating sentence probabilities of sentence_i in a sentences_log_p matrix (M_models x S_sentences)
        args:
            sentence_i_log_p (numpy.array) M_models x S_sentences
        returns:
            loss (numpy.array) M_sentences
        """
        
        M_designs, S_sentences=sentence_i_log_p.shape
        
        # copy the sentence_log_p matrix M_designs times to form an M_designs x M_models x S_sentences tensor
        modified_sentences_log_p=np.repeat(np.expand_dims(sentences_log_p,0),M_designs,axis=0)
        
        # paste sentence_i_log_p (log probabilities of potential replacement sentences for each model)
        modified_sentences_log_p[:,:,i_sentence]=sentence_i_log_p
        
        loss=loss_func(modified_sentences_log_p)        
        return loss
    
    return loss_fun_i
                          


def optimize_sentence_set(n_sentences,models,loss_fun,sent_len=8,
                         initial_sampling='uniform',external_stopping_check=lambda: False,
                         internal_stopping_condition=lambda: False,
                         start_with_identical_sentences=True, max_steps=10000):
    """ Optimize a sentence set of n_sentences.
    n_sentences (int) how many sentences to optimize (e.g., 2 for a sentence pair).
    models (list) a list of model objects
    response_models (list) a matching list of ModelToHumanDecision instances (or None)
    loss_fun (function) given a ( models x sentences) ... ?
    sentence_len (int) how many words
    initial_sampling (str) 'uniform' or 'proportional'
    start_with_identical_sentences (boolean) 
    returns a results dict, or False (if aborted due to external_stopping_check)
    max_steps (int)
    """
    
    # initializes a list (wordis) that determines which word is replaced at each step.
    # the order is designed to be cyclical
    def get_word_replacement_order(sent_len,max_steps):
        wordi=np.arange(sent_len)                                           
        wordis=[]
        while len(wordis)<max_steps:
            random.shuffle(wordi)
            wordis=wordis+list(wordi)
        return wordis
    wordis=get_word_replacement_order(sent_len,max_steps):
    
    # initialize random sentences
    def initialize_sentence(sent_len,initial_sampling='uniform'):
        if initial_sampling == 'uniform':
            vocab_low_freqs1=np.ones([len(vocab_low_freqs)])/len(vocab_low_freqs)
            vocab_cap_freqs1=np.ones([len(vocab_cap_freqs)])/len(vocab_cap_freqs)
        elif initial_sampling == 'proportional':
            vocab_low_freqs1=vocab_low_freqs
            vocab_cap_freqs1=vocab_cap_freqs
        else:
            raise ValueError('unsupported initial_sampling argument')
        words=list(np.random.choice(vocab_cap, 1, p=vocab_cap_freqs1)) + list(np.random.choice(vocab_low, sent_len-1, p=vocab_low_freqs1, replace=False))
        sent=' '.join(words)
        return sent1
    if start_with_identical_sentences:
        sentences=[initialize_sentence(sent_len,initial_sampling='uniform')]*n_sentences
    else:
        sentences=[initialize_sentence(sent_len,initial_sampling='uniform') for i_sent in range(n_sentences)]
            
    # get initial sentence probabilities
    def get_sentence_log_probabilities(models,sentences):
        """ Return a (model x sentence) log_probability numpy matrix """
        log_p=np.nan(len(models),len(sentences))
        for i_model, model in enumerate(models):
            for i_sentence, sentence in enumerate(sentences):
                log_p[i_model,i_sentence]=model.sent_prob(sentence)
        return log_p
                
    # sentence_log_p[i,j] is the log-probabilitiy assigned to sentence j by model i.                                           
    sentences_log_p=get_sentence_log_probabilities(models,sentences)
    if response_models is not None:
        decision_log_p=sentence_log_p_to_decision_log_p(sentences_log_p,response_models)
        current_loss=loss((M_designs, N_sentences, K_models)

    
    #TODO - call verbose fun
#     if verbose>=2:
#         print('\n')
#         print('initialized sentence '+str(num_sents) +', model: '+model1_name+', step: '+str(step_ind))
#         print('Target: '+str(step))
#         print('Current: '+str(model1_sent1_prob))
#         print(sent1)

    n_consequetive_failed_replacements=0 # (previsouly 'cycle' - this keeps track on how many words we failed to replace since the last succesful word replacement)
    termination_reason=''
    for step in range(max_steps):
        if external_stopping_check():
            # abort optimization (e.g., another worker completed the sentence).
            return False

        if internal_stopping_condition(loss):
            termination_reason='internal_stopping_condition'
            break
        elif cycle==sent_len:
            termination_reason='converged'
            break
            
#         if np.abs(model1_sent1_prob-step) < 1:
#             exclusive_write_line(fname,sent1+'.',max_lines=max_sentences)
#             print('target log-prob achieved, sentence added to file.')
#             break
        
        # determine which word are we trying to replace at this step.
        wordi=int(wordis[step])

        if wordi==0: # use capitalized words for the first word.
            vocab=vocab_cap
        else:
            vocab=vocab_low
        
        # for a given word placement, change one sentence at a time (using a random order).
        sentence_indecis_to_modify=list(range(n_sentences))
        random.shuffle(sentence_indecis_to_modify)
        for i_sentence in sentence_indecis_to_modify:
            sentence = sentences[i_sentence]
            words = sentence.split(' ') # break the sentences into words
            cur_word=words[wordi] # current word to replace
            
            # get word-based log-probs for potential replacement of words[wordi], from each model.
            # some of the models return exact sentence-log-probability for each potential replacement word,
            # others return an approximation.
            
            all_models_word_df=None
            models_with_approximate_probs=[]
            for i_model, model in enumerate(models):

                # get a list of potential replacement words. For each word, we have the corresponding sentence log probability.
                word_list, model_word_probs, prob_type = get_word_prob_from_model(model,words,wordi)
                
                model_words_df=pd.DataFrame(index=word_list)
                if prob_type=='exact': 
                    model_words_df['exact_'+str(i_model)]=model_word_probs
                    model_words_df['approximate_'+str(i_model)]=np.nan
                else:
                    model_words_df['approximate_'+str(i_model)]=model_word_probs
                    model_words_df['exact_'+str(i_model)]=np.nan
                    models_with_approximate_probs.append(i_model)
                
                # make sure the exact probability for the current word is included (we already have it from the previous round)
                model_words_df.at[cur_word,'exact_'+str(i_model)]=p[i_model,i_sentence]
                
                # accumulate replacement words and associated probabilities across models
                if all_models_word_df is None:
                    all_models_word_df=model_words_df
                else:
                    all_models_word_df=all_models_word_df.join(model_words_df,how='inner') # this keeps only the intersecion of the word_lists!
                    # (if we have neither approximate nor exact log-prob of a word for one of the models, we don't consider it).
            
            # For models with approximate log probabilities, we need
            # at least two datapoints to fit a linear regression from
            # approximate log probs to exact log probs.
            #
            # We'll evaluate the exact probabilties for the words with the maximal
            # and minimal approximate probabilities for these models.            
            for i_model in models_with_approximate_probs:
                words_to_evaluate=[all_models_word_df['approximate_'+str(i_model)].idxmax(),all_models_word_df['approximate_'+str(i_model)].idxmin()]
                for word_to_evaluate in words_to_evaluate:
                    if not np.isnan(all_models_word_df.at[word_to_evaluate,'exact_'+str(i_model)]):
                        continue # don't waste time evaluating the word if we already have its exact log prob.
                        # (this might happen if the max probability word is also the current word).                    
                    modified_words=words.copy()
                    modified_words[wordi]=word_to_evaluate
                    modified_sent=' '.join(modified_words)
                    modified_sent_prob=models[i_model].get_model1_sent_prob(modified_sent)
                    
                    all_models_word_df.at[word_to_evaluate,'exact_'+str(i_model)]=modified_sent_prob
          
            # define loss function for updating sentence_i, with the other sentences fixed.
            loss_fun_i=get_loss_fun_with_respect_to_sentence_i(sentences_log_p,i_sentence,loss_func,response_models=response_models)
                        
            opt=SetInterpolationSearch(loss_fun=loss_fun_i,g=g,
                initial_observed_xs=initial_observed_xs,initial_observed_ys=initial_observed_ys,
                initial_xs_guesses=initial_xs_guesses,
                h_method='LinearRegression')
            
            # word replacement loop                    
            
            # get minimal loss for words with exact probability
            
            # if 
            
            # get the minimal loss for all words (using approximate probabilities)
            
            cur_loss=loss_fun(model1_sent1_prob)
            n_words_evaluated_without_loss_improvement=0
            max_n_words_to_consider_without_loss_improvement=50
            found_useful_replacement=False
            best_loss_so_far=np.inf
            best_word1_model1_sent1t_prob=None
            best_word1=None
            word_iteration=0
            while (not found_useful_replacement) and (n_words_evaluated_without_loss_improvement<max_n_words_to_consider_without_loss_improvement):
                word_iteration+=1

                if word_iteration>1 and not found_useful_replacement:
                    evaluation_word_idx,_,_=opt.get_unobserved_loss_minimum()
                    if evaluation_word_idx is None:
                        if verbose>=3:
                            print('word replacements exhausted.')
                        break
                    word1=word1_list[evaluation_word_idx]

                    # evaluate the sentence log-probability with the next word
                    words1t=words1o.copy()
                    words1t[wordi]=word1
                    sent1t=' '.join(words1t)
                    model1_sent1t_prob=get_model1_sent_prob(sent1t)

                    # update the optimizer
                    opt.update_query_result(xs=[evaluation_word_idx],ys=[model1_sent1t_prob])

                    if verbose>=3:
                        print("evaluated: {:<30} | log-prob: {:06.1f}→ {:06.1f} | ".format(
                            cur_word1+'→ '+word1,model1_sent1_prob,model1_sent1t_prob,),end='')

                # check the updated, observed global minimum
                minimum_loss_word_idx,minimum_loss=opt.get_observed_loss_minimum()

                if minimum_loss is not None:

                    if minimum_loss<best_loss_so_far: # track relative improvement (used for stopping criterion)
                        best_word1=word1_list[minimum_loss_word_idx]
                        best_word1_model1_sent1t_prob=opt.ys[minimum_loss_word_idx].item()
                        best_loss_so_far=minimum_loss
                        n_words_evaluated_without_loss_improvement=0

                        if minimum_loss<cur_loss: # absolute improvement - stop word-level search
                            found_useful_replacement=True
                    else:
                        n_words_evaluated_without_loss_improvement+=1

                    if verbose>=3:
                            print("best replacement: {:<30} | log-prob: {:06.1f}→ {:06.1f} | loss: {:06.1f}→ {:06.1f}".format(
                                cur_word1+'→ '+best_word1,model1_sent1_prob,best_word1_model1_sent1t_prob,cur_loss,best_loss_so_far,)
                                ,end='')
                            if n_words_evaluated_without_loss_improvement>0:
                                print(" | {:02d} words w/o loss improvement.".format(n_words_evaluated_without_loss_improvement))
                            else:
                                print("")

            # matplotlib plot of sentence probabilities as function of word probabilities
            # if n_words_evaluated_without_loss_improvement>5:
            #     opt.debugging_figure()

        if found_useful_replacement:
            new_word1=best_word1
            new_word1o=new_word1

            words1[wordi]=new_word1.upper()

            sent1p=' '.join(words1)

            words1[wordi]=new_word1.lower()
            if wordi==0 or new_word1o[0].isupper():
                words1[wordi]=new_word1.lower().capitalize()

            sent1=' '.join(words1)

            # this evaluation is superfluous because best_word1_model1_sent1t_prob
            # already has the probability of the current sentence
            # this is included here as a sanity check
            model1_sent1_prob=get_model1_sent_prob(sent1)

            # this is faster:
#                    model1_sent1_prob=best_word1_model1_sent1t_prob
            assert(np.isclose(model1_sent1_prob,best_word1_model1_sent1t_prob,rtol=1e-3))

            if verbose>=2:
                print('Target: '+str(step))
                print('Current: '+str(model1_sent1_prob))
                print(sent1p)
        else:
            if verbose>=2:
                print('no useful replacement for ' + cur_word1 +  ' (a total of {} possible sentences considered.)'.format(len(opt.fully_observed_obs())))
            cycle+=1
            
    # loop completed (or terminated), organize results.
    if termination_reason=='':
        termination_reason='max_steps'    
    results={'sentences':sentences,
             'sentences_log_p':sentences_log_p,
             'loss':loss,
             'step',step,
             'termination_reason':termination_reason}
    return results
            
if model_loaded:
del model1
del get_model1_sent_prob
del get_model1_word_probs
del steps
model_loaded=False
