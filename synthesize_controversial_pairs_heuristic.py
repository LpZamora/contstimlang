import os, pickle
import itertools, random
import csv

import torch
import numpy as np

# can we move these out of main?
from lstm_class import RNNLM
from bilstm_class import RNNLM_bilstm
from rnn_class import RNNModel

from model_functions import model_factory
from sentence_optimization import optimize_sentence_set, initialize_random_word_sentence
from utils import exclusive_write_line, get_n_lines, hash_dict

def synthesize_controversial_sentence_pair(all_model_names,results_csv_folder=None,sent_len=8,max_pairs=10,
                                           allow_only_prepositions_to_repeat=False, natural_initialization=True,
                                           verbose=3):
    n_sentences=2 # we optimize a pair of sentences

    sentences_to_change=[1] # change the second sentence

    if allow_only_prepositions_to_repeat:
        allowed_repeating_words=set(pickle.load(open("preps.pkl","rb")))
        keep_words_unique=True
    else:
        allowed_repeating_words=None
        keep_words_unique=False

    with open('sents10k.txt') as f:
        natural_sentences=[l.strip().rstrip('.') for l in f]

    for model_name_pair in itertools.product(all_model_names,repeat=2):
        [model1_name, model2_name] = model_name_pair
        if model1_name == model2_name:
            continue

        if results_csv_folder is not None:
            results_csv_fname=os.path.join(results_csv_folder,model1_name+'_vs_'+model2_name+'.csv')

            # This function halts the sentence optimization if the file while its running
            def is_file_complete():
                is_complete=get_n_lines(results_csv_fname)>=max_pairs
                if is_complete and verbose>=3:
                    print("found "+str(get_n_lines(results_csv_fname)) + " lines in " + results_csv_fname)
                return is_complete

            if is_file_complete(): #check right away
                continue

            external_stopping_check=is_file_complete # the check will be used within the optimization function
        else: # no results saving, just display.
            external_stopping_check=lambda: False

        # allocate GPUs
        model_GPU_IDs=[]
        cur_GPU_ID=0
        for model_name in model_name_pair:
            model_GPU_IDs.append(cur_GPU_ID)
            if not model_name in ['bigram', 'trigram']: # bigram and trigram models run on CPU, so gpu_id will be ignored
                cur_GPU_ID+=1
                if cur_GPU_ID>=torch.cuda.device_count():
                    cur_GPU_ID=0

        # load models
        models=[]
        for model_name, model_GPU_ID in zip(model_name_pair, model_GPU_IDs):
            print("loading "+ model_name + " into gpu " + str(model_GPU_ID) + '...',end='')
            models.append(model_factory(model_name,model_GPU_ID))
            print("done.")

        while not is_file_complete():

            def loss_func(sentences_log_p):
                """
                Given a reference natural sentence, form a variant of it which is at least as likely according to one model,
                and as unlikely as possible according to the other.

                args:
                sentences_log_p (numpy.array, D_designs x M_models x S_sentences, or M_models x S_sentences) current log probabilities,
                such that sentences_log_p[m,s]=log_p(sentence_s|model_m

                comment: both M_models and S_sentences should be 2
                """

                # we'd like to push down the log-probability assigned by model 2 to s2 (the optimized sentence) as much as possible:
                m2s1=sentences_log_p[...,1,0]
                m2s2=sentences_log_p[...,1,1]
                l=m2s2-m2s1 #m2s1 is actually constant, we subtract it so l=0 for sentences with identical log-prob.

                # this penalty is activated when model 1 assigns lower log-probability to s2 (the optimized sentence) compared to s1 (the reference sentence):
                m1s1=sentences_log_p[...,0,0]
                m1s2=sentences_log_p[...,0,1]
                p=np.maximum(m1s1-m1s2,0.0)

                return l + 1e5 * p # we care more about satisfying the constraints than decreasing the loss

            def monitoring_func(sentences,sentences_log_p):
                print(model1_name+":"+'{:.2f}/{:.2f}'.format(sentences_log_p[...,0,0],sentences_log_p[...,0,1]))
                print(model2_name+":"+'{:.2f}/{:.2f}'.format(sentences_log_p[...,1,0],sentences_log_p[...,1,1]))

            internal_stopping_condition=lambda loss: False # don't stop optimizing until convergence

            if natural_initialization:
                initial_sentences=[random.choice(natural_sentences)]*n_sentences
            else:
                initial_sentences=[random.choice(natural_sentences),
                initialize_random_word_sentence(sent_len,initial_sampling='uniform')]

            results=optimize_sentence_set(n_sentences,models=models,loss_func=loss_func,sentences=initial_sentences,sent_len=sent_len,
                                     initial_sampling='uniform',external_stopping_check=external_stopping_check,
                                     monitoring_func=monitoring_func,
                                     internal_stopping_condition=internal_stopping_condition,
                                     start_with_identical_sentences=True, max_steps=10000,
                                     keep_words_unique=keep_words_unique,
                                     allowed_repeating_words=allowed_repeating_words,
                                     sentences_to_change=sentences_to_change,
                                     verbose=verbose)
            if results is False: # optimization was terminated
                continue

            sentences=results['sentences']
            sentences_log_p=results['sentences_log_p']
            print(sentences)
            monitoring_func(sentences,sentences_log_p)

            if results['loss']<0:
                # sentence 1, sentence 2, loss, model_1_log_prob_sent1, model_1_log_prob_sent2, model_2_log_prob_sent1, model_2_log_prob_sent2,
                outputs=results['sentences']+[results['loss']]+list(sentences_log_p.flat)
                line=','.join(map(str, outputs))
                exclusive_write_line(results_csv_fname,line,max_pairs)
            else:
                print("failed optimization, not writing results.")

if __name__ == "__main__":
    #all_model_names=['bigram','trigram','rnn','lstm','gpt2','bert','bert_whole_word','roberta','xlm','electra','bilstm']

    all_model_names=['bigram','trigram','gpt2']
    sent_len=8

    results_csv_folder=os.path.join('synthesized_sentences',
                                    '20210126_controverisal_sentence_pairs_heuristic_random_init',
                                     '{}_word'.format(sent_len))

    synthesize_controversial_sentence_pair(all_model_names,results_csv_folder=results_csv_folder,
                                           sent_len=sent_len,
                                           allow_only_prepositions_to_repeat=True,
                                           natural_initialization=False,
                                           max_pairs=10,verbose=3)
