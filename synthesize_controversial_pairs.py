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
from sentence_optimization import optimize_sentence_set, human_choice_controversiality_loss, human_choice_controversiality_loss_log_scale, human_choice_probability, controversiality_score
from model_to_human_decision_torch import load_decision_model, Naive
from utils import exclusive_write_line, get_n_lines, hash_dict

def synthesize_controversial_sentence_pair(all_model_names,decision_models_folder,
                                           results_csv_folder=None,sent_len=8,max_pairs=10,
                                           allow_only_prepositions_to_repeat=False,natural_initialization=False,verbose=3):
    n_sentences=2 # we optimize a pair of sentences
    
    if allow_only_prepositions_to_repeat:
        allowed_repeating_words=set(pickle.load(open("preps.pkl","rb")))
        keep_words_unique=True
    else:
        allowed_repeating_words=None
        keep_words_unique=False
        
    if natural_initialization:
        with open('sents10k.txt') as f:
            natural_sentences=[l.strip().rstrip('.') for l in f]
    
    for model_name_pair in itertools.combinations(all_model_names,2):
        [model1_name, model2_name] = model_name_pair
        
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

        # load human decision models
        optimizer='LBFGS'
        human_choice_response_models=[]
        for model_name in model_name_pair:
            path = os.path.join(decision_models_folder,model_name+'.pkl')
            human_choice_response_models.append(load_decision_model(path,device='cpu'))

        model_prior=None # assume flat prior over models for MI calculation
        
        while not is_file_complete():                          

            def loss_func(sentences_log_p):
                return human_choice_controversiality_loss_log_scale(sentences_log_p,human_choice_response_models=human_choice_response_models,model_prior=model_prior)

            def monitoring_func(sentences,sentences_log_p):
                human_p=human_choice_probability(sentences_log_p,human_choice_response_models=human_choice_response_models,log_scale=False)    
                print(model1_name+":"+'{:.2f}/{:.2f}'.format(human_p[0,0,0],human_p[0,0,1]))
                print(model2_name+":"+'{:.2f}/{:.2f}'.format(human_p[0,1,0],human_p[0,1,1]))

            internal_stopping_condition=lambda loss: False # don't stop optimizing until convergence

            if natural_initialization:
                initial_sentences=[random.choice(natural_sentences)]*n_sentences
            else:
                initial_sentences=None
                
            results=optimize_sentence_set(n_sentences,models=models,loss_func=loss_func,sentences=initial_sentences,sent_len=sent_len,
                                     initial_sampling='uniform',external_stopping_check=external_stopping_check,
                                     monitoring_func=monitoring_func,
                                     internal_stopping_condition=internal_stopping_condition,
                                     start_with_identical_sentences=True, max_steps=10000,
                                     keep_words_unique=keep_words_unique,     
                                     allowed_repeating_words=allowed_repeating_words,                                          
                                     verbose=verbose)
            if results is False: # optimization was terminated
                continue
            sentences=results['sentences']
            sentences_log_p=results['sentences_log_p']
            print(sentences)
            monitoring_func(sentences,sentences_log_p)

            # write results to file:                                    
            human_p = human_choice_probability(sentences_log_p,human_choice_response_models=human_choice_response_models,log_scale=False)
            MI = controversiality_score(human_p)
            
            if not np.isclose(MI,0):
                # sentence 1, sentence 2, MI, model_1_log_prob_sent1, model_1_log_prob_sent2, model_2_log_prob_sent1, model_2_log_prob_sent2,
                # model_1_human_prob_sent1, model_1_human_prob_sent2, model_2_human_prob_sent1, model_2_human_prob_sent2
                outputs=results['sentences']+[MI]+list(sentences_log_p.flat)+list(human_p.flat)                        
                line=','.join(map(str, outputs))             
                exclusive_write_line(results_csv_fname,line,max_pairs)
            else:
                print("MI=0, not writing result.")
                          
if __name__ == "__main__":
    #all_model_names=['bigram','trigram','rnn','lstm','gpt2','bert','bert_whole_word','roberta','xlm','electra','bilstm']
    all_model_names=['bigram','trigram','rnn','lstm','gpt2']
    #all_model_names=['bigram','trigram','gpt2']
    sent_len=8
                          
    optimizer='LBFGS'
    decision_model_class='FixedWidthSquashing'
    decision_models_folder=os.path.join('decision_models',
                                        '20210118_10th_percentile_squash',decision_model_class+'_' +optimizer + '_{}_word'.format(sent_len))
    
    results_csv_folder=os.path.join('synthesized_sentences',
                                    '20210118_controverisal_sentence_pairs_no_reps_natural_init_10th_percentile_squash',
                                    decision_model_class+'_' +optimizer + '_{}_word'.format(sent_len))
    
    synthesize_controversial_sentence_pair(all_model_names,decision_models_folder,
                                           results_csv_folder=results_csv_folder,
                                           sent_len=sent_len,
                                           allow_only_prepositions_to_repeat=True,natural_initialization=True,
                                           max_pairs=4,verbose=3)
