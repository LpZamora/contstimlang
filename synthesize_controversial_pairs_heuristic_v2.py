import os, pickle
import itertools, random
import csv
import random


import torch
import numpy as np
import pandas as pd

# can we move these out of main?
from lstm_class import RNNLM
from bilstm_class import RNNLM_bilstm
from rnn_class import RNNModel

from model_functions import model_factory
from sentence_optimization import optimize_sentence_set, initialize_random_word_sentence
from utils import exclusive_write_line, get_n_lines, hash_dict

from task_scheduler import TaskScheduler


class NaturalSentenceAssigner():
    def __init__(self,all_model_names,seed = 42):
        with open('sents_for_expt2_optimization_full_list.txt') as f:
            natural_sentences=[l.strip().rstrip('.') for l in f]

        natural_sentences = pd.DataFrame({'sentence':natural_sentences})
        natural_sentences = natural_sentences.sample(frac=1,random_state=42) # shuffle sentences

        n_total_sentences=len(natural_sentences)

        model_pairs = list(itertools.combinations(all_model_names,2))
        model_pairs = [tuple(sorted(pair)) for pair in model_pairs]

        random.Random(seed).shuffle(model_pairs)

        n_model_pairs = len(model_pairs)

        sentence_groups=natural_sentences.groupby(np.arange(len(natural_sentences)) % n_model_pairs)

        self.all_model_names = all_model_names
        self.model_pairs = model_pairs
        self.model_pair_dict = {tuple(model_pair):sentence_group[1].sort_index() for model_pair, sentence_group in zip(model_pairs,sentence_groups)}

    def get_sentences(self, model_pair):
        return self.model_pair_dict[tuple(sorted(model_pair))]


def synthesize_controversial_sentence_pair(all_model_names, natural_sentence_assigner, results_csv_folder=None,sent_len=8,
                                           allow_only_prepositions_to_repeat=False, max_pairs = 5, natural_initialization = True,
                                           direction='down',verbose=3):

    n_sentences=2 # we optimize a pair of sentences

    sentences_to_change=[1] # change the second sentence

    sch=TaskScheduler(max_job_time_in_seconds=3600*6)
    job_df = sch.to_pandas()

    try:
        job_id_df = pd.DataFrame(list(job_df['job_id']))
    except:
        job_id_df = None

    # determine which model pair has least completed jobs or running
    model_pairs_stats=[]
    for model_name_pair in itertools.product(all_model_names,repeat=2):
        [model1_name, model2_name] = model_name_pair

        if model1_name == model2_name:
            continue

        if job_id_df is not None and len(job_id_df)>0:
            n_jobs=((job_id_df['model_1']==model1_name) & (job_id_df['model_2']==model2_name)).sum()
        else:
            n_jobs=0

        model_pairs_stats.append({'model_1':model1_name,'model_2':model2_name,'n_jobs':n_jobs,'tie_breaker':random.random()})
    model_pairs_stats=pd.DataFrame(model_pairs_stats)
    model_pairs_stats=model_pairs_stats.sort_values(by=['n_jobs','tie_breaker'],ascending=True)

    # list model pairs, models with less sentences first
    model_pair_list = list(zip(model_pairs_stats['model_1'],model_pairs_stats['model_2']))

    if allow_only_prepositions_to_repeat:
        allowed_repeating_words=set(pickle.load(open("preps.pkl","rb")))
        keep_words_unique=True
    else:
        allowed_repeating_words=None
        keep_words_unique=False

    for model_name_pair in model_pair_list:
        [model1_name, model2_name] = model_name_pair

        if results_csv_folder is not None:
            results_csv_fname=os.path.join(results_csv_folder,model1_name+'_vs_'+model2_name+'.csv')
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

        models_loaded=False

        n_optimized = 0

        natural_sentence_df = sentence_assigner.get_sentences(model_name_pair)
        for i_natural_sentence, (sentence_index, natural_sentence) in enumerate(zip(natural_sentence_df.index,natural_sentence_df['sentence'])):

            if i_natural_sentence >= 101:
                break

            # use all sentences for all model pairs
            job_id = {'natural_sentence':natural_sentence, 'model_1':model1_name,'model_2':model2_name}

            # use each natural sentence exactly twice (for the same model pair, using different roles)
            #sorted_model_name_pair=sorted(model_pair)
            #job_id = {'natural_sentence':natural_sentence, 'sorted_model_name_pair':sorted_model_name_pair}

            success = sch.start_job(job_id)
            if not success:
                continue

            print('optimizing sentence {} ({}) for {} vs {}'.format(i_natural_sentence, sentence_index, model1_name, model2_name))
            if not models_loaded: # load models
                models=[]
                for model_name, model_GPU_ID in zip(model_name_pair, model_GPU_IDs):
                    print("loading "+ model_name + " into gpu " + str(model_GPU_ID) + '...',end='')
                    models.append(model_factory(model_name,model_GPU_ID))
                    print("done.")
                    models_loaded=True

            if direction == 'down':
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
            elif direction =='up':
                def loss_func(sentences_log_p):
                    """
                    Given a reference low-probability sentence, form a variant of it which is as at least as unlikely according to one model,
                    and as likely as possible according to the other.

                    args:
                    sentences_log_p (numpy.array, D_designs x M_models x S_sentences, or M_models x S_sentences) current log probabilities,
                    such that sentences_log_p[m,s]=log_p(sentence_s|model_m

                    comment: both M_models and S_sentences should be 2
                    """

                    # we'd like to push up the log-probability assigned by model 2 to s2 (the optimized sentence) as much as possible:
                    m2s1=sentences_log_p[...,1,0]
                    m2s2=sentences_log_p[...,1,1]
                    l=m2s1-m2s2 #m2s1 is actually constant, we add it so l=0 for sentences with identical log-prob.

                    # this penalty is activated when model 1 assigns higher log-probability to s2 (the optimized sentence) compared to s1 (the reference sentence):
                    m1s1=sentences_log_p[...,0,0]
                    m1s2=sentences_log_p[...,0,1]
                    p=np.maximum(m1s2-m1s1,0.0)

                    return l + 1e5 * p # we care more about satisfying the constraints than decreasing the loss

            def monitoring_func(sentences,sentences_log_p):
                print(model1_name+":"+'{:.2f}/{:.2f}'.format(sentences_log_p[...,0,0],sentences_log_p[...,0,1]))
                print(model2_name+":"+'{:.2f}/{:.2f}'.format(sentences_log_p[...,1,0],sentences_log_p[...,1,1]))

            internal_stopping_condition=lambda loss: False # don't stop optimizing until convergence

            if natural_initialization:
                initial_sentences=[natural_sentence]*n_sentences
            else:
                initial_sentences=[initialize_random_word_sentence(sent_len,initial_sampling='uniform')]*n_sentences

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

            # sentence 1, sentence 2, loss, model_1_log_prob_sent1, model_1_log_prob_sent2, model_2_log_prob_sent1, model_2_log_prob_sent2,
            outputs=[sentence_index]+results['sentences']+[results['loss']]+list(sentences_log_p.flat)
            line=','.join(map(str, outputs))
            exclusive_write_line(results_csv_fname,line)
            sch.job_done(job_id,results=results)

            n_optimized+=1
            if n_optimized >= max_pairs:
                break


if __name__ == "__main__":
    all_model_names=['bigram','trigram','rnn','lstm','gpt2','bert','roberta','xlm','electra']

    sentence_assigner=NaturalSentenceAssigner(all_model_names)
    sent_len=8

    results_csv_folder=os.path.join('synthesized_sentences',
                                     '20210224_controverisal_sentence_pairs_heuristic_natural_init_allow_rep',
                                      '{}_word'.format(sent_len))

    synthesize_controversial_sentence_pair(all_model_names,sentence_assigner,results_csv_folder=results_csv_folder,
                                           sent_len=sent_len,
                                           allow_only_prepositions_to_repeat=True,
                                           natural_initialization=True, direction='down',
                                           max_pairs=5,verbose=3)
