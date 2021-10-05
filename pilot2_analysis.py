# %%
from collections import OrderedDict
import random
import re
import itertools
import math
import copy
import os, pathlib

from contextlib import contextmanager

import numpy as np
import pandas as pd
from pandas.core.reshape.reshape import get_dummies
from tqdm import tqdm
import pandas as pd
import scipy.stats
import statsmodels.stats.multitest
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import seaborn as sns
from matplotlib.gridspec import GridSpec

from metroplot import metroplot



# filename definitions
natural_controversial_sentences_fname = 'sents_reddit_natural_June2021_selected.csv'
synthetic_controversial_sentences_fname = 'synthesized_sentences/20210224_controverisal_sentence_pairs_heuristic_natural_init_allow_rep/8_word_9_models_100_sentences_per_pair_best10.csv'
#from isotonic_response_model import overfitted_isotonic_mapping

model_order=['gpt2','roberta','electra','bert','xlm','lstm','rnn','trigram','bigram']

# define model colors
# color pallete from colorbrewer2 : https://colorbrewer2.org/?type=qualitative&scheme=Set1&n=9#type=qualitative&scheme=Set1&n=9
# model_palette={'gpt2':'#e41a1c','roberta':'#377eb8','electra':'#4daf4a','bert':'#984ea3','xlm':'#ff7f00','lstm':'#ffff33','rnn':'#a65628','trigram':'#f781bf','bigram':'#999999'}

# color pallete from colorbrewer2 : https://colorbrewer2.org/?type=qualitative&scheme=Paired&n=9#type=qualitative&scheme=Accent&n=3
model_palette={'gpt2':'#a6cee3','roberta':'#1f78b4','electra':'#b2df8a','bert':'#33a02c','xlm':'#fb9a99','lstm':'#e31a1c','rnn':'#fdbf6f','trigram':'#ff7f00','bigram':'#cab2d6'}

# nice labels for the models
model_name_dict = {'gpt2':'GPT-2',
                   'roberta':'RoBERTa',
                   'electra':'ELECTRA',
                   'bert':'BERT',
                   'xlm':'XLM',
                   'lstm':'LSTM',
                   'rnn':'RNN',
                   'trigram':'3-gram',
                   'bigram':'2-gram'}

# https://stackoverflow.com/questions/38629830/how-to-turn-off-autoscaling-in-matplotlib-pyplot
@contextmanager
def autoscale_turned_off(ax=None):
  ax = ax or plt.gca()
  lims = [ax.get_xlim(), ax.get_ylim()]
  yield
  ax.set_xlim(*lims[0])
  ax.set_ylim(*lims[1])


# transform lists and dicts to use nice model labels
def niceify(x):
     if isinstance(x,list):
          return [model_name_dict[m] for m in x]
     elif isinstance(x,pd.core.series.Series):
          return x.apply(lambda model: model_name_dict[model])
     elif isinstance(x,dict):
          return {model_name_dict[k]:v for k,v in x.items()}
     elif isinstance(x,str):
          return model_name_dict[x]
     else:
          raise ValueError

def align_sentences(df):
     """ To ease analysis, we align all trials so the order of sentences
     within each sentence pair is lexicographical rather than based on display position.
     This ensures that different subjects can be directly compared to each other.

     This script also changes creates a numerical "rating" column, with 1 = strong preference for sentence1, 6 = strong preference for sentence2.

     """

     fields_to_del = ['Unnamed: 0', 'Zone Type']
     df2=[]
     for i_trial, old_trial in df.iterrows():
          new_trial=OrderedDict()

          flip_required = old_trial.sentence1>old_trial.sentence2

          for col in df.columns:
               if col in fields_to_del:
                    continue
               elif col=='Zone Name':
                    rating=float(old_trial['Zone Name'].replace('resp',''))
                    if flip_required:
                         rating=7-rating
                    new_trial['rating']=rating
               else:
                    reference_col=col
                    if flip_required:
                         if 'sentence1' in col:
                              reference_col = col.replace('sentence1','sentence2')
                         elif 'sentence2' in col:
                              reference_col = col.replace('sentence2','sentence1')
                    new_trial[col]=old_trial[reference_col]

          if flip_required:
               new_trial['sentence1_location']='right'
          else:
               new_trial['sentence1_location']='left'

          df2.append(new_trial)

     df2=pd.DataFrame(df2)
     return df2

def recode_model_targeting(df):
     """ create readable model targeting labels.

     The follow fields are added to df:

     sentence1_model_targeted_to_accept - the model that was optimized to view sentence1 as at least as likely as a natural sentence
     sentence1_model_targeted_to_reject - the model that was optimized to view sentence1 as unlikely
     (+the equivalent fields for sentence2)


     returns:
     modified datafrate
     """

     df = df.copy()

     natural_controversial_sentences_df = pd.read_csv(natural_controversial_sentences_fname)
     synthetic_controversial_sentences_df = pd.read_csv(synthetic_controversial_sentences_fname)

     def get_natural_controversial_sentence_targeting(sentence, natural_controversial_sentences_df):
          match_sentence1 = natural_controversial_sentences_df[natural_controversial_sentences_df.sentence1 == sentence]
          match_sentence2 = natural_controversial_sentences_df[natural_controversial_sentences_df.sentence2 == sentence]
          if len(match_sentence1)==1 and len(match_sentence2) == 0:
               sentence_model_targeted_to_accept = match_sentence1['model_2'].item()
               sentence_model_targeted_to_reject = match_sentence1['model_1'].item()
          elif len(match_sentence1)==0 and len(match_sentence2) == 1:
               sentence_model_targeted_to_accept = match_sentence2['model_1'].item()
               sentence_model_targeted_to_reject = match_sentence2['model_2'].item()
          else:
               raise Exception
          return sentence_model_targeted_to_accept, sentence_model_targeted_to_reject

     def get_synthetic_controversial_sentence_targeting(sentence, synthetic_controversial_sentences_df):
          match_sentence1 = synthetic_controversial_sentences_df[synthetic_controversial_sentences_df.S1 == sentence]
          match_sentence2 = synthetic_controversial_sentences_df[synthetic_controversial_sentences_df.S2 == sentence]
          if len(match_sentence1)==1 and len(match_sentence2) == 0:
               sentence_model_targeted_to_accept = match_sentence1['m1'].item()
               sentence_model_targeted_to_reject = match_sentence1['m2'].item()
          elif len(match_sentence1)==0 and len(match_sentence2) == 1:
               sentence_model_targeted_to_accept = match_sentence2['m2'].item()
               sentence_model_targeted_to_reject = match_sentence2['m1'].item()
          else:
               raise Exception
          return sentence_model_targeted_to_accept, sentence_model_targeted_to_reject

     for idx_trial, trial in tqdm(df.iterrows(), total=len(df), desc = 'recoding model targeting'):

          if {trial.sentence1_type, trial.sentence2_type} == {'N1','N2'}:
               # Natural controversial sentences. Here we make sure the model predictions are fully crossed.
               df.loc[idx_trial,'trial_type']='natural_controversial'

               # go back to original CSV and grab model targeting info
               for s in [1,2]:
                    df.loc[idx_trial,f'sentence{s}_model_targeted_to_accept'], df.loc[idx_trial,f'sentence{s}_model_targeted_to_reject'] = get_natural_controversial_sentence_targeting(getattr(trial,f'sentence{s}'), natural_controversial_sentences_df)

               # sanity check for model predictions
               model_A_s1 = df.loc[idx_trial,'sentence1_model_targeted_to_accept']
               model_R_s1 = df.loc[idx_trial,'sentence1_model_targeted_to_reject']
               model_A_s2 = df.loc[idx_trial,'sentence2_model_targeted_to_accept']
               model_R_s2 = df.loc[idx_trial,'sentence2_model_targeted_to_reject']
               assert (model_A_s1 == model_R_s2) and (model_R_s1 == model_A_s2)
               p_A_s1 = getattr(trial,f'sentence1_{model_A_s1}_prob')
               p_R_s1 = getattr(trial,f'sentence1_{model_R_s1}_prob')
               p_A_s2 = getattr(trial,f'sentence2_{model_A_s2}_prob')
               p_R_s2 = getattr(trial,f'sentence2_{model_R_s2}_prob')
               assert (p_A_s1 > p_R_s2) & (p_A_s2 > p_R_s1)

          elif {trial.sentence1_type, trial.sentence2_type} == {'N','S1'} or {trial.sentence1_type, trial.sentence2_type} == {'N','S2'}:
               # A synthetic controversial sentence vs. a natural sentence.
               df.loc[idx_trial,'trial_type']='natural_vs_synthetic'
               n = [1,2][[trial.sentence1_type, trial.sentence2_type].index('N')]
               s = [2,1][[trial.sentence1_type, trial.sentence2_type].index('N')]

               # go back to original CSV and grab model targeting info
               df.loc[idx_trial,f'sentence{s}_model_targeted_to_accept'], df.loc[idx_trial,f'sentence{s}_model_targeted_to_reject'] = get_synthetic_controversial_sentence_targeting(getattr(trial,f'sentence{s}'), synthetic_controversial_sentences_df)

               # sanity check for model predictions
               model_A_s = df.loc[idx_trial,f'sentence{s}_model_targeted_to_accept']
               model_R_s = df.loc[idx_trial,f'sentence{s}_model_targeted_to_reject']
               p_A_s = getattr(trial,f'sentence{s}_{model_A_s}_prob')
               p_R_s = getattr(trial,f'sentence{s}_{model_R_s}_prob')
               p_A_n = getattr(trial,f'sentence{n}_{model_A_s}_prob')
               p_R_n = getattr(trial,f'sentence{n}_{model_R_s}_prob')
               assert (p_A_s>=p_A_n) and (p_R_s<p_R_n)
          elif {trial.sentence1_type, trial.sentence2_type} == {'S1','S2'}:
               # Synthetic controversial sentence vs. Synthetic controversial sentence
               df.loc[idx_trial,'trial_type']='synthetic_vs_synthetic'

               # go back to original CSV and grab model targeting info
               for s in [1,2]:
                    df.loc[idx_trial,f'sentence{s}_model_targeted_to_accept'], df.loc[idx_trial,f'sentence{s}_model_targeted_to_reject'] = get_synthetic_controversial_sentence_targeting(getattr(trial,f'sentence{s}'), synthetic_controversial_sentences_df)

               # sanity check for model predictions
               model_A_s1 = df.loc[idx_trial,'sentence1_model_targeted_to_accept']
               model_R_s1 = df.loc[idx_trial,'sentence1_model_targeted_to_reject']
               model_A_s2 = df.loc[idx_trial,'sentence2_model_targeted_to_accept']
               model_R_s2 = df.loc[idx_trial,'sentence2_model_targeted_to_reject']
               assert (model_A_s1 == model_R_s2) and (model_R_s1 == model_A_s2)
               p_A_s1 = getattr(trial,f'sentence1_{model_A_s1}_prob')
               p_R_s1 = getattr(trial,f'sentence1_{model_R_s1}_prob')
               p_A_s2 = getattr(trial,f'sentence2_{model_A_s2}_prob')
               p_R_s2 = getattr(trial,f'sentence2_{model_R_s2}_prob')
               assert (p_A_s1 > p_R_s2) & (p_A_s2 > p_R_s1)
          elif {trial.sentence1_type, trial.sentence2_type} == {'C1','C2'}:
               # Catch trials (natural sentences and their shuffled version)
               df.loc[idx_trial,'trial_type']='natural_vs_shuffled'
          elif {trial.sentence1_type, trial.sentence2_type} == {'R1','R2'}:
               # randomly sampled natural sentences
               df.loc[idx_trial,'trial_type']='randomly_sampled_natural'
          else:
               raise ValueError

          # remove 1 and 2 from sentence type
          df.loc[idx_trial,'sentence1_type']=trial.sentence1_type.replace('1','').replace('2','')
          df.loc[idx_trial,'sentence2_type']=trial.sentence2_type.replace('1','').replace('2','')
     return df


def add_leave_one_subject_predictions(df):
     """ Leave one subject out noise ceiling
     All of the following measures are lower bounds on the noise ceiling.
     In other words, an ideal model should be at least as good as these measures.
     """

     # The LOOSO loop.
     df2=df.copy()

     df2['binarized_choice_probability_NC_LB']=np.nan
     df2['binarized_choice_probability_NC_UB']=np.nan
     df2['majority_vote_NC_LB']=np.nan
     df2['majority_vote_NC_UB']=np.nan
     df2['mean_rating_NC_LB']=np.nan
     df2['mean_rating_NC_UB']=np.nan

     # def assign(df, index, field,val):
     #      df.iloc[index,df.columns.get_loc(field)]=val
     #      print(df.iloc[index,df.columns.get_loc(field)])

     for trial_idx, trial in tqdm(df.iterrows(),total=len(df),desc='leave one subject out NC calculation.'):
          # choose all trials with the same sentence pair in OTHER subjects.
          mask=(df['sentence_pair']==trial['sentence_pair']) \
               & (df['subject']!=trial['subject'])
          reduced_df=df[mask]

          # we add three kinds of noise ceiling:

          # 1. binarized choice probability:
          # the predicted probability that a subject will prefer sentence2
          # (to be used for binomial likelihood evaluation)
          df2.loc[trial_idx,'binarized_choice_probability_NC_LB']=(reduced_df['rating']>=4).mean()

          # 2. simple majority vote (1: sentence2, 0: sentence1)
          # to be used for accuracy evaluation)
          if df2.loc[trial_idx,'binarized_choice_probability_NC_LB']>0.5:
               df2.loc[trial_idx,'majority_vote_NC_LB']=1
          elif df2.loc[trial_idx,'binarized_choice_probability_NC_LB']<0.5:
               df2.loc[trial_idx,'majority_vote_NC_LB']=0
          else:
               raise Warning(f'Tied predictions for trial {trial_idx}. Randomzing prediction.')
               df2.loc[trial_idx,'majority_vote_NC_LB']=random.choice([0,1])

          # 3. And last, we simply average the ratings
          # to be used for correlation based measures
          df2.loc[trial_idx,'mean_rating_NC_LB']=(reduced_df['rating']).mean()


     for trial_idx, trial in tqdm(df.iterrows(),total=len(df),desc='upper bound NC calculation.'):
          # choose all trials with the same sentence pair in ALL subjects.
          mask=(df['sentence_pair']==trial['sentence_pair'])
          reduced_df=df[mask]

          # 1. binarized choice probability:
          # the predicted probability that a subject will prefer sentence2
          # (to be used for binomial likelihood evaluation)
          df2.loc[trial_idx,'binarized_choice_probability_NC_UB']=(reduced_df['rating']>=4).mean()

          # 2. simple majority vote (1: sentence2, 0: sentence1)
          # to be used for accuracy evaluation)
          if df2.loc[trial_idx,'binarized_choice_probability_NC_UB']>0.5:
               df2.loc[trial_idx,'majority_vote_NC_UB']=1
          elif df2.loc[trial_idx,'binarized_choice_probability_NC_UB']<0.5:
               df2.loc[trial_idx,'majority_vote_NC_UB']=0
          else:
               # print(f'Tied predictions for trial {trial_idx}. Randomizing prediction.')
               df2.loc[trial_idx,'majority_vote_NC_UB']=random.choice([0,1])

          # 3. And last, we simply average the ratings
          # to be used for correlation based measures
          df2.loc[trial_idx,'mean_rating_NC_UB']=(reduced_df['rating']).mean()
          # (note - this is not a true upper bound on the noise ceiling for average model-subject correlation coefficient)

     return df2

def filter_trials(df,targeted_model=None,targeting=None,trial_type=None):
     """ subsets a trial dataframe.

     one can filter by trial type, as well as by the targeted model.
     for trial_type='natural_vs_synthetic', we can also specify targeting='accept'|'reject',
     to select only trials in which the synthetic sentence was optimized to be accepted/rejected
     by targeted_model.

     args:
     targeted_model (str) which model was targeted
     targeting (str) 'accept'|'reject'|None for all. what kind of targeting.
     trial_type (str) 'natural_controversial'|'natural_vs_synthetic'|'synthetic_vs_synthetic'|'natural_vs_shuffled'|'randomly_sampled_natural'| None for all

     returns reduced df
     """

     mask = df['subject']==df['subject'] # all True series

     if trial_type is not None:
          mask = mask & (df['trial_type']==trial_type)

     if targeted_model is None:
          assert targeting is None, 'targeting should only be specified when targeted_model is specified'
     elif targeting is None:
          # we keep the trial if one of the sentences targeted the model
          mask = mask & (
                    (df['sentence1_model_targeted_to_accept']==targeted_model) |
                    (df['sentence1_model_targeted_to_reject']==targeted_model) |
                    (df['sentence2_model_targeted_to_accept']==targeted_model) |
                    (df['sentence2_model_targeted_to_reject']==targeted_model)
          )
     elif targeting is 'accept':
          assert trial_type == 'natural_vs_synthetic', 'filtering trials by accept/reject targeting only makes sense for N vs S trials.'
          mask = mask & (
                    (df['sentence1_model_targeted_to_accept']==targeted_model) |
                    (df['sentence2_model_targeted_to_accept']==targeted_model)
          )
     elif targeting is 'reject':
          assert trial_type == 'natural_vs_synthetic', 'filtering trials by accept/reject targeting only makes sense for N vs S trials.'
          mask = mask & (
                    (df['sentence1_model_targeted_to_reject']==targeted_model) |
                    (df['sentence2_model_targeted_to_reject']==targeted_model)
          )
     else:
          raise ValueError
     return df.copy()[mask]

def get_models(df):
     """ a helper function for extracting model names from column names """
     models = [re.findall('sentence1_(.+)_prob',col)[0] for col in df.columns if re.search('sentence1_(.+)_prob',col)]
     return models


def group_level_signed_ranked_test(reduced_df, models):
     group_level_df = reduced_df.groupby('subject_group').mean()
     results=[]
     for model1, model2 in itertools.combinations(models,2):
          s, p = scipy.stats.wilcoxon(group_level_df[model1],group_level_df[model2],zero_method='zsplit')
          results.append({'model1':model1,'model2':model2,'p-value':p,
          'avg_model1_minus_avg_model2':(group_level_df[model1]-group_level_df[model2]).mean()})

     # noise ceiling comparisons
     for model1 in models:
          if 'NC_LB' in group_level_df.columns:
               model2 = 'NC_LB'
          elif ('NC_LB_'+ model1) in group_level_df.columns:
               model2 = 'NC_LB_' + model1
          s, p = scipy.stats.wilcoxon(group_level_df[model1],group_level_df[model2],zero_method='zsplit')
          results.append({'model1':model1,'model2':model2,'p-value':p,
               'avg_model1_minus_avg_model2':(group_level_df[model1]-group_level_df[model2]).mean()})
     results = pd.DataFrame(results)
     _ , results['FDR_corrected_p-value']=statsmodels.stats.multitest.fdrcorrection(results['p-value'])
     return results

def calc_binarized_accuracy(df):

     df2=df.copy()
     models = get_models(df)

     """ binarizes model and human predictions and returns 1 or 0 for prediction correctness """
     for model in models:

          assert not (df['sentence2_'+model + '_prob']==df['sentence1_'+model + '_prob']).any(), f'found tied prediction for model {model}'
          model_predicts_sent2=df['sentence2_'+model + '_prob']>df['sentence1_'+model + '_prob']
          human_chose_sent2=df['rating']>=4

          # store trial-level accuracy
          df2[model]=(model_predicts_sent2==human_chose_sent2).astype('float')

          # drop probability
          df2=df2.drop(columns=['sentence1_'+model + '_prob','sentence2_'+model + '_prob'])

     df2['NC_LB']=(df2['majority_vote_NC_LB']==human_chose_sent2).astype(float)
     df2['NC_UB']=(df2['majority_vote_NC_UB']==human_chose_sent2).astype(float)
     return df2

def build_all_html_files(df):
     models = get_models(df)
     for model1 in models:
          for model2 in models:
               if model1==model2:
                    continue
               build_html_file(df,os.path.join('result_htmls',model1 + "_vs_" +model2 + ".html" ),model1,model2)
               print('.')

def build_html_file(df, filepath, model1, model2):
     """ Generate HTML files with trials organized by sentence triplets """
     triplets = organize_pairwise_data_into_triplets(df,model1,model2)

     # for sorting the triplets, we calcuate triplet-level accuracy for model 1
     triplet_level_accuracy = (
           triplets['h_N_NS1']/(triplets['h_N_NS1']+triplets['h_S1_NS1'])
          +triplets['h_S2_NS2']/(triplets['h_N_NS2']+triplets['h_S2_NS2'])
          +triplets['h_S2_S1S2']/(triplets['h_S1_S1S2']+triplets['h_S2_S1S2']))/3

     triplets['model_1_accuracy']=triplet_level_accuracy

     ind = (-triplet_level_accuracy).argsort()
     triplets = triplets.loc[ind]

     with open('triplet_html_table_template.html','r') as f:
          template = f.read()

     html = '\
<!DOCTYPE html>\n\
<html>\n\
<head>\n\
\t<meta name="viewport" content="width=device-width, initial-scale=1">\n\
</head>\n\
<body>\n'

     for i_triplet, triplet in triplets.iterrows():
          new_entry = copy.copy(template)
          for k, v in triplet.items():
               if k.startswith('p_') and isinstance(v,float):
                    str_v = f'{v:.1f}'
               elif k.startswith('h_') and k.endswith('_NS1'):
                    total = triplet['h_N_NS1']+triplet['h_S1_NS1']
                    str_v=f'{round(v):d}/{round(total):d}'
               elif k.startswith('h_') and k.endswith('_NS2'):
                    total = triplet['h_N_NS2']+triplet['h_S2_NS2']
                    str_v=f'{round(v):d}/{round(total):d}'
               elif k.startswith('h_') and k.endswith('_S1S2'):
                    total = triplet['h_S1_S1S2']+triplet['h_S2_S1S2']
                    str_v=f'{round(v):d}/{round(total):d}'
               elif k.startswith('model') and k.endswith('_name'):
                    str_v = niceify(v)
               else:
                    str_v=f'{v}'
               new_entry = new_entry.replace(k,str_v)
          html+=new_entry
          html+='\n<br>\n'
     html+='\n</body>\n</head>\n'

     with open(filepath,'w') as f:
          template = f.write(html)

def organize_pairwise_data_into_triplets(df,model1,model2):
     """ for a pair of model, return all N vs. S and S vs. S trials organized in triplets """
     models = get_models(df)

     # get only N-vs-S or S-vs-S trials in which the two models were targeted
     df2=df[(
               (
                    ((df['sentence1_model']==model1) & (df['sentence2_model']==model2)) |
                    ((df['sentence1_model']==model2) & (df['sentence2_model']==model1))
               ) &
                     (df['trial_type'].isin(['natural_vs_synthetic','synthetic_vs_synthetic']))
               )]

     # reduce subjects
     df2 = df2.assign(humans_chose_sentence2=(df2['rating']>=4).astype(float))
     df2 = df2.assign(humans_chose_sentence1=(df2['rating']<=3).astype(float))
     df2 = df2.drop(columns=[f'sentence1_{m}_prob' for m in models if (m not in {model1,model2})])
     df2 = df2.drop(columns=[f'sentence2_{m}_prob' for m in models if (m not in {model1,model2})])
     df2 = df2.drop(columns=['subject','Trial Number','Reaction Time'])
     df2 = df2.groupby(['sentence_pair','sentence1','sentence2','sentence1_model','sentence2_model',
                        'sentence1_model_targeted_to_accept','sentence2_model_targeted_to_accept',
                        'sentence1_model_targeted_to_reject','sentence2_model_targeted_to_reject',
                        'sentence1_type','sentence2_type','trial_type'],dropna=False,
                        ).sum().reset_index()

     # further split trials to sub-types
     df3_N_vs_S_model1_targeted_to_reject = df2[
          ((df2['sentence1_model_targeted_to_reject']==model1) & (df2['sentence2_type']=='N')) |
          ((df2['sentence2_model_targeted_to_reject']==model1) & (df2['sentence1_type']=='N'))
          ]

     df3_N_vs_S_model2_targeted_to_reject = df2[
          ((df2['sentence1_model_targeted_to_reject']==model2) & (df2['sentence2_type']=='N')) |
          ((df2['sentence2_model_targeted_to_reject']==model2) & (df2['sentence1_type']=='N'))
          ]

     df3_S_vs_S = df2[df2['trial_type']=='synthetic_vs_synthetic']

     # these three groups should togheter consist the original set of trials
     assert len(pd.concat([df3_N_vs_S_model1_targeted_to_reject,df3_N_vs_S_model2_targeted_to_reject,df3_S_vs_S]))==len(df2)

     # build triplets dataframe
     triplets=[]
     for i_trial, trial in df3_N_vs_S_model1_targeted_to_reject.iterrows():
          cur_triplet=dict()
          cur_triplet['model1_name']=model1
          cur_triplet['model2_name']=model2
          if trial['sentence1_type']=='N':
               cur_triplet['NATURAL_SENTENCE']=trial['sentence1']
               cur_triplet['SYNTHETIC_SENTENCE_1']=trial['sentence2']
               cur_triplet['p_N_m1'] = trial['sentence1_'+model1+'_prob']
               cur_triplet['p_N_m2'] = trial['sentence1_'+model2+'_prob']
               cur_triplet['p_S1_m1'] = trial['sentence2_'+model1+'_prob']
               cur_triplet['p_S1_m2'] = trial['sentence2_'+model2+'_prob']
               cur_triplet['h_N_NS1'] = trial['humans_chose_sentence1']
               cur_triplet['h_S1_NS1'] = trial['humans_chose_sentence2']
          elif trial['sentence2_type']=='N':
               cur_triplet['NATURAL_SENTENCE']=trial['sentence2']
               cur_triplet['SYNTHETIC_SENTENCE_1']=trial['sentence1']
               cur_triplet['p_N_m1'] = trial['sentence2_'+model1+'_prob']
               cur_triplet['p_N_m2'] = trial['sentence2_'+model2+'_prob']
               cur_triplet['p_S1_m1'] = trial['sentence1_'+model1+'_prob']
               cur_triplet['p_S1_m2'] = trial['sentence1_'+model2+'_prob']
               cur_triplet['h_N_NS1'] = trial['humans_chose_sentence2']
               cur_triplet['h_S1_NS1'] = trial['humans_chose_sentence1']
          else:
               raise ValueError

          # find the other S vs N trial (with the model roles flipped)
          other_trial=df3_N_vs_S_model2_targeted_to_reject[(
               (df3_N_vs_S_model2_targeted_to_reject['sentence1']==cur_triplet['NATURAL_SENTENCE']) |
               (df3_N_vs_S_model2_targeted_to_reject['sentence2']==cur_triplet['NATURAL_SENTENCE'])
               )]
          assert len(other_trial)==1
          other_trial=other_trial.iloc[0]

          if other_trial['sentence1_type']=='N':
               cur_triplet['SYNTHETIC_SENTENCE_2']=other_trial['sentence2']
               cur_triplet['p_S2_m1'] = other_trial['sentence2_'+model1+'_prob']
               cur_triplet['p_S2_m2'] = other_trial['sentence2_'+model2+'_prob']
               cur_triplet['h_N_NS2'] = other_trial['humans_chose_sentence1']
               cur_triplet['h_S2_NS2'] = other_trial['humans_chose_sentence2']
          elif other_trial['sentence2_type']=='N':
               cur_triplet['SYNTHETIC_SENTENCE_2']=other_trial['sentence1']
               cur_triplet['p_S2_m1'] = other_trial['sentence1_'+model1+'_prob']
               cur_triplet['p_S2_m2'] = other_trial['sentence1_'+model2+'_prob']
               cur_triplet['h_N_NS2'] = other_trial['humans_chose_sentence2']
               cur_triplet['h_S2_NS2'] = other_trial['humans_chose_sentence1']
          else:
               raise ValueError

          # and now the corresponding S vs S trial
          other_trial=df3_S_vs_S[(
               ((df3_S_vs_S['sentence1']==cur_triplet['SYNTHETIC_SENTENCE_1']) & (df3_S_vs_S['sentence2']==cur_triplet['SYNTHETIC_SENTENCE_2'])) |
               ((df3_S_vs_S['sentence1']==cur_triplet['SYNTHETIC_SENTENCE_2']) & (df3_S_vs_S['sentence2']==cur_triplet['SYNTHETIC_SENTENCE_1']))
               )]
          assert len(other_trial)==1
          other_trial=other_trial.iloc[0]
          if other_trial['sentence1']==cur_triplet['SYNTHETIC_SENTENCE_1']:
               cur_triplet['h_S1_S1S2']=other_trial['humans_chose_sentence1']
               cur_triplet['h_S2_S1S2']=other_trial['humans_chose_sentence2']
          elif other_trial['sentence2']==cur_triplet['SYNTHETIC_SENTENCE_1']:
               cur_triplet['h_S1_S1S2']=other_trial['humans_chose_sentence2']
               cur_triplet['h_S2_S1S2']=other_trial['humans_chose_sentence1']
          else:
               raise ValueError
          triplets.append(cur_triplet)
     return pd.DataFrame(triplets)

def reduce_within_model(df, reduction_func, models=None, trial_type=None, targeting=None):
     """ group data by targeted model and then apply reduction_func within each group """
     if models is None:
          models = get_models(df)
     results =[]
     for model in models:
          # drop trials in which the model was not targeted
          filtered_df = filter_trials(df, targeted_model=model, targeting = targeting, trial_type = trial_type)

          # drop the probabilities of the other models
          filtered_df = filtered_df.drop(columns=[f'sentence1_{m}_prob' for m in models if (m != model)])
          filtered_df = filtered_df.drop(columns=[f'sentence2_{m}_prob' for m in models if (m != model)])

          # reduce (e.g., calculate accuracy, correlation)
          reduced_df = reduction_func(filtered_df)

          if 'NC_LB' in reduced_df.columns:
               reduced_df=reduced_df.rename(columns={'NC_LB':'NC_LB_'+model})
          if 'NC_UB' in reduced_df.columns:
               reduced_df=reduced_df.rename(columns={'NC_UB':'NC_UB_'+model})

          results.append(reduced_df)
     results = pd.concat(results)
     return results


def model_specific_performace_dot_plot(df, models, ylabel='% accuracy',title=None,ax=None, each_dot_is='subject_group',chance_level=None, model_specific_NC=False, pairwise_sig=None, tick_label_fontsize=8):

     matplotlib.rcParams.update({'font.size': 10})
     matplotlib.rcParams.update({'font.family':'sans-serif'})
     matplotlib.rcParams.update({'font.sans-serif':'Arial'})

     if ax is None:
          plt.figure(figsize=(4.5,2.5))
          ax=plt.gca()
     else:
          plt.sca(ax)

     # rearrange the data
     if each_dot_is == 'subject':
          reduced_df = df.groupby('subject').mean()
     elif each_dot_is == 'sentence_pair':
          reduced_df = df.groupby('sentence_pair').mean()
     elif each_dot_is == 'subject_group':
          reduced_df = df.groupby('subject_group').mean()

#     from behavioral_data_analysis.figure_settings import model_color_dict
     long_df=pd.melt(reduced_df.reset_index(),id_vars=[each_dot_is],var_name = 'model', value_vars=models,value_name='prediction_accuracy')

     long_df['model_label'] = niceify(long_df['model'])

#    strippplot
     g1=sns.stripplot(data=long_df, y='model_label',x='prediction_accuracy',hue='model_label',linewidth=0.333,edgecolor='white',jitter=0.25,alpha=1,size=4,zorder=2,palette=niceify(model_palette),order=niceify(model_order))
     g1.legend_.remove()

     # it seems that stripplot can't produce markers with outlines
     verts = [(-1,-4.8),(-1,4.8),(1,4.8),(1,-4.8),(-1,-4.8)]
     g2=sns.stripplot(data=long_df.groupby('model_label').mean().reset_index(), y='model_label',x='prediction_accuracy',hue='model_label',edgecolors='k',linewidth=0.5,jitter=0.0,alpha=1,size=15,zorder=3,palette=niceify(model_palette),order=niceify(model_order), marker=verts)
     g2.legend_.remove()

     # # bootstrapped violin
     # bootstrapped_df=[]
     # for i_bootstrap in range(1000):
     #      bs_sample = reduced_df.sample(n=len(reduced_df), replace=True).mean()
     #      bs_sample['i_bootstrap']=i_bootstrap
     #      bootstrapped_df.append(bs_sample)
     # bootstrapped_df = pd.DataFrame(bootstrapped_df)
     # long_bootstrapped_df=pd.melt(bootstrapped_df.reset_index(),id_vars=['i_bootstrap'],var_name = 'model', value_vars=models,value_name='prediction_accuracy')

     # g1=sns.violinplot(data=long_bootstrapped_df, y='model',x='prediction_accuracy',hue='model',zorder=2,palette=model_palette,order=model_order, width=1.0, dodge=False)
     # g1.legend_.remove()

     # g1=sns.violinplot(data=long_df, y='model',x='prediction_accuracy',hue='model',zorder=2,palette=model_palette,order=model_order)
     # g1.legend_.remove()

     plt.xlim([0.0,1.0])
     plt.xticks([0,0.25,0.5,0.75,1.0],['0','25%','50%','75%','100%'])
     ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
     ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)

     if chance_level is not None:
          plt.axvline(x=chance_level,linestyle='--',color='k',zorder=-100)

     def plot_NC_sig(NC_LB,i_model,model,model_specific_NC):
          if model_specific_NC:
               NC_LB_fieldname='NC_LB_' + model
          else:
               NC_LB_fieldname='NC_LB'
          row_filter = (
                         ((pairwise_sig['model1']==model) & (pairwise_sig['model2']==NC_LB_fieldname)) |
                         ((pairwise_sig['model2']==model) & (pairwise_sig['model1']==NC_LB_fieldname))
                    )
          assert row_filter.sum()==1, f'expecting to find a single comparison of model {model} to the noise ceiling'
          p_value = pairwise_sig[row_filter].iloc[0].loc['FDR_corrected_p-value']
          if p_value<0.05:
               mask = pd.notnull(df[model]) # consider only lines with non-null performance value
               model_score=df[mask][model].mean()
               assert NC_LB > model_score, f'model score for {model} is greater than the lower bound on the noise ceiling, this is not correctly represented with the asterisks scheme.'
               #plt.text(NC_LB, i_model,'*', ha='center',color='k')
               plt.plot(NC_LB,i_model,marker=(5, 2, 0),color='k',markersize=8,zorder=1000)

     # model-specific noise ceiling
     with autoscale_turned_off(ax):
          for i_model, model in enumerate(model_order):
               if model_specific_NC:
                    NC_LB = df['NC_LB_'+model].mean()
                    NC_UB = df['NC_UB_'+model].mean()
                    ax.add_patch(matplotlib.patches.Rectangle(xy=(NC_LB,i_model-0.4),width=NC_UB-NC_LB, height=0.8,alpha=1.0,fill=True,edgecolor='silver',facecolor='silver',linewidth=1.0, zorder=-100))
               else:
                    NC_LB = df['NC_LB'].mean()
                    NC_UB = df['NC_UB'].mean()
               plot_NC_sig(NC_LB,i_model,model,model_specific_NC=model_specific_NC)
          if not model_specific_NC:
               # plot single rectangle for all models.
               ax.add_patch(matplotlib.patches.Rectangle(xy=(NC_LB,-1),width=NC_UB-NC_LB, height=len(models)-1+2.0,alpha=1.0,fill=True,edgecolor='silver',facecolor='silver',linewidth=1.0, zorder=-100))
     return ax

def plot_one_main_results_panel(df, reduction_fun, models, cur_panel_cfg, ax = None, metroplot_ax = None, chance_level=None, metroplot_preallocated_positions=None, tick_label_fontsize=8):
     """ plot one panel of model-human alignment dot plot """
     if cur_panel_cfg['only_targeted_trials']:
          # analyze each model's performance on the the trials that targeted it.
          reduced_df = reduce_within_model(df, reduction_func=reduction_fun, models=models, trial_type=cur_panel_cfg['trial_type'], targeting=cur_panel_cfg['targeting'])
     else: # don't filter trials by targeted model
          # filter trials by trial_type
          filtered_df = filter_trials(df, trial_type = cur_panel_cfg['trial_type'], targeted_model=None, targeting = None)
          # reduce (e.g., calculate accuracy, correlation)
          reduced_df = reduction_fun(filtered_df)

     pairwise_sig = group_level_signed_ranked_test(reduced_df,models)

     print(cur_panel_cfg['title'])
     print(reduced_df.mean())
     print(pairwise_sig)

     model_specific_performace_dot_plot(reduced_df,models, ylabel='% accuracy',title=None,ax=ax, each_dot_is='subject_group', chance_level=chance_level, model_specific_NC=cur_panel_cfg['only_targeted_trials'], pairwise_sig=pairwise_sig, tick_label_fontsize=tick_label_fontsize)

     if metroplot_ax is not None: # plot metroplot significance plot
          level_to_location={model_name:i for i, model_name in enumerate(model_order)}
          # prepare dataframe for metroplot
          plots_df = pairwise_sig.rename(columns={'model1':'level1','model2':'level2'})
          plots_df['effect_direction']=np.sign(pairwise_sig['avg_model1_minus_avg_model2'])
          plots_df['is_sig']=pairwise_sig['FDR_corrected_p-value']<0.05

          # to make metrplots aligned across panels, we fix the xlim of the metroplot axes
          if metroplot_preallocated_positions is not None:
               element_axis_lim=[-0.4,0.25+metroplot_preallocated_positions]
          else:
               element_axis_lim=None

          metroplot(plots_df,level_to_location,metroplot_element_order=model_order,level_axis='y',ax=metroplot_ax,
                                   dominating_effect_direction=1,level_pallete=model_palette, level_axis_lim=ax.get_ylim(),
                                   element_axis_lim=element_axis_lim,
                                   empty_dot_fill_color='w',
                                   marker='o', linewidth=0.5, markeredgewidth=0.5, markersize=8)


def plot_main_results_figures(df, models=None, plot_metroplot=True, save_folder = None):
     if models is None:
          models = get_models(df)

     reduction_fun = calc_binarized_accuracy

     # define figure structure
     panel_cfg  = [
          {'title':'natural vs. shuffled',                        'only_targeted_trials':False,     'trial_type':'natural_vs_shuffled',         'targeting':None,},
          {'title':'randomly sampled natural sentences',          'only_targeted_trials':False,     'trial_type':'randomly_sampled_natural',    'targeting':None,},
          {'title':'natural controversial sentences',             'only_targeted_trials':True,      'trial_type':'natural_controversial',       'targeting':None,},
          {'title':'p(synthetic|model ) <  p(natural|model)',                    'only_targeted_trials':True,      'trial_type':'natural_vs_synthetic',        'targeting':'reject'},
          {'title':'p(synthetic|model ) ≥ p(natural|model)',                     'only_targeted_trials':True,      'trial_type':'natural_vs_synthetic',        'targeting':'accept'},
          {'title':'synthetic vs. synthetic',                     'only_targeted_trials':True,      'trial_type':'synthetic_vs_synthetic',      'targeting':None,},
          {'title':'all trials',                                  'only_targeted_trials':False,     'trial_type':None,                          'targeting':None,},
     ]

     figure_plans = [
          {'panels':[0,1,2], 'fname':'natural_and_natural_controversial.pdf'},
          {'panels':[3,4,5], 'fig_size':(7,7), 'fname':'synthetic.pdf'},
          {'panels':[6], 'fig_size':(7,7/3), 'fname':'all_trials.pdf'},
     ]

     # all of the following measures are in inches
     top_margin = 0.0
     bottom_margin = 0.2
     left_margin = 0
     right_margin = 0.00
     panel_h = 1.8
     panel_w = 1.8
     v_space_above_panel = 0.225
     v_space_below_panel = 0.25
     h_space1 = 0 # horizontal space between left column and result panel
     h_space2 = 0.05 # horizontal space between result column and metroplot
     metroplot_w = 1.125
     metroplot_preallocated_positions=7 # how many significance elements are we expecting.
     fig_w = 7 # total figure width - 7 inches (PNAS limitation, Nat. Comm is more generous)
     left_col_w = fig_w - (left_margin+h_space1+panel_w+h_space2+metroplot_w+right_margin)
     panel_title_fontsize = 10
     axes_label_fontsize = 10
     tick_label_fontsize = 8

     figs =[]
     for figure_plan in figure_plans:

          # setup gridspec grid structure
          n_panels = len(figure_plan['panels'])

          widths_in_inches = [left_margin, left_col_w, h_space1, panel_w, h_space2, metroplot_w, right_margin]
          horizontal_elements = [None,None,None,'panel',None,'metroplot',None]
          assert np.isclose(np.sum(widths_in_inches),fig_w), 'widths don''t match'

          heights_in_inches = [top_margin]
          vertical_elements =[None]
          for i_panel in range(n_panels):
               heights_in_inches.extend([v_space_above_panel,panel_h,v_space_below_panel])
               vertical_elements.extend([None,f'panel{i_panel}',None])
          heights_in_inches.append(bottom_margin)
          vertical_elements.append(None)
          fig_h = np.sum(heights_in_inches) # height is set adaptively

          print(f"figure {figure_plan['fname']} size: {fig_w},{fig_h} inches")

          fig = plt.figure(figsize=(fig_w,fig_h))

          gs0=GridSpec(ncols=len(widths_in_inches), nrows=len(heights_in_inches), figure=fig,
          width_ratios=widths_in_inches,height_ratios=heights_in_inches, hspace=0, wspace=0,top=1,bottom=0,left=0,right=1)

          for i_panel, panel_idx in enumerate(figure_plan['panels']):
               cur_panel_cfg = panel_cfg[panel_idx]
               result_panel_ax=fig.add_subplot(gs0[vertical_elements.index(f'panel{i_panel}'), horizontal_elements.index('panel')])
               if plot_metroplot:
                    metroplot_ax = fig.add_subplot(gs0[vertical_elements.index(f'panel{i_panel}'), horizontal_elements.index('metroplot')])
               else:
                    metroplot_ax = None
               plot_one_main_results_panel(df, reduction_fun, models, cur_panel_cfg, ax=result_panel_ax, chance_level=0.5, metroplot_ax=metroplot_ax, metroplot_preallocated_positions=metroplot_preallocated_positions, tick_label_fontsize=tick_label_fontsize)
               result_panel_ax.set_title(cur_panel_cfg['title'],fontdict={'fontsize':panel_title_fontsize})
               result_panel_ax.set_ylabel('')
               if i_panel == n_panels-1:
                    result_panel_ax.set_xlabel('human-choice prediction accuracy',fontdict={'fontsize':axes_label_fontsize})
               else:
                    result_panel_ax.set_xlabel('')
          if save_folder is not None:
               pathlib.Path(save_folder).mkdir(parents=True,exist_ok=True)
               fig.savefig(os.path.join(save_folder,figure_plan['fname']), dpi=600)
               figs.append(fig)
     return figs

def pairwise_binary_choice_analysis(df):

     # first, let's consider only trials with N, S sentence pairs
     mask = (df['sentence1_type']=='N') | (df['sentence2_type']=='N')
     reduced_df2 = df[mask]

     models = get_models(df)
     n_models = len(models)

     n = np.zeros((n_models,n_models)) # n[i,j] is the count of trials in which the i-th model preferred the natural sentence and the j-th model the synthetic sentence.
     x = np.zeros((n_models,n_models)) # x[i,j] is the count of trials "" and the human participant favor the natural sentence as well.

     for idx_trial, trial in reduced_df2.iterrows():
          if trial.sentence1_type=='N' and trial.sentence2_type=='S':
               i_model = models.index(trial.sentence1_model)
               j_model = models.index(trial.sentence2_model)
               if trial.rating<=3:
                    x[i_model,j_model]+=1
          elif trial.sentence1_type=='S' and trial.sentence2_type=='N':
               i_model = models.index(trial.sentence2_model)
               j_model = models.index(trial.sentence1_model)
               if trial.rating>=4:
                    x[i_model,j_model]+=1
          else:
               raise ValueError

          n[i_model,j_model]+=1
     print (n)
     print (x)
     fig=plt.figure()
     ax=plt.gca()
     im = ax.imshow(x/n)
     plt.colorbar(im, ax=ax)

     ax.set_xticks(np.arange(n_models))
     ax.set_yticks(np.arange(n_models))
     ax.set_xticklabels(models)
     ax.set_yticklabels(models)
     #ax.xaxis.tick_top()
     plt.ylabel('models targeted to reject the synthetic sentence P(S)<P(N)')
     plt.xlabel('models held equivariant P(S)>=P(N)')
     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
     plt.tight_layout()
     plt.show()
def log_prob_pairs_to_scores(df, transformation_func='diff'):
     """ each model predicts pair of log-probabilities. To predict human ratings, this function convert each pair to a scalar score """
     if transformation_func == 'diff': # The most naive approach - use the difference of log probabilities as predictor
          transformation_func = lambda lp_1, lp_2: lp_2-lp_1
     elif transformation_func == 'rand': # random detereministic transformation. just for debugging
          transformation_func = lambda lp_1, lp_2: [np.random.RandomState(int(-(p1+p2)*10+42)).rand() for p1,p2 in zip(lp_1, lp_2)]
     # elif transformation_func == 'isotonic': # best possible mapping
     #      transformation_func = lambda lp_1, lp_2: overfitted_isotonic_mapping(lp_1,lp_2,df['rating'])
     else:
          raise ValueError
     models = get_models(df)
     df2=df.copy()
     for model in models:
          df2[model]=transformation_func(df['sentence1_'+model + '_prob'],df['sentence2_'+model + '_prob'])
          df2=df2.drop(columns=['sentence1_'+model + '_prob','sentence2_'+model + '_prob'])
     return df2



def get_correlation_based_accuracy_old(df,models, correlation_func='pearsonr'):
     """ within each subject, correlate model scalar scores with subject's ratings"""
     if correlation_func == 'pearsonr':
          correlation_func = lambda rating, score: scipy.stats.pearsonr(rating,score)[0]
     elif correlation_func == 'spearmanr':
          correlation_func = lambda rating, score: scipy.stats.spearmanr(rating,score,nan_policy='omit')[0]
     elif correlation_func == 'kendall-Tau-b':
          # scipy.stats nightly has a somersd function, but it's not yet officially released
          correlation_func = lambda rating, score: scipy.stats.kendalltau(rating,score,method='asymptotic',variant='b')[0]

     # calculate within-subject correlations
     df2=[]
     for subject in df['Participant Private ID'].unique():
          cur_subject_results={'Participant Private ID':subject}
          mask=df['Participant Private ID']==subject
          reduced_df=df[mask]
          for model in models:
               cur_subject_results[model]=correlation_func(reduced_df['rating'],reduced_df[model])
          df2.append(cur_subject_results)
     df2=pd.DataFrame(df2)
     return df2

def get_correlation_based_accuracy(df,models, correlation_func='pearsonr',targeting=None, pair_type=None):
     """ within each subject, correlate model scalar scores with subject's ratings"""
     if correlation_func == 'pearsonr':
          correlation_func = lambda rating, score: scipy.stats.pearsonr(rating,score)[0]
     elif correlation_func == 'spearmanr':
          correlation_func = lambda rating, score: scipy.stats.spearmanr(rating,score,nan_policy='omit')[0]
     elif correlation_func == 'kendall-Tau-b':
          # scipy.stats nightly has a somersd function, but it's not yet officially released
          correlation_func = lambda rating, score: scipy.stats.kendalltau(rating,score,method='asymptotic',variant='b')[0]

     df = df.copy()
     df['pair_type']=['N_vs_S' if t1=='N' or t2=='N' else 'S_vs_S' for t1,t2 in zip(df.sentence1_type, df.sentence2_type)]

     df2=[]

     results = []
     for model in models:
          cur_result={'model':model}
          # split according to model targeting
          if targeting is None:
               mask = np.ones(len(df),dtype=bool)
          elif targeting == 'targeted':
               mask = (df.sentence1_model == model) | (df.sentence2_model == model)
               cur_result['targeting']='targeted'
          elif targeting == 'untargeted':
               mask =  (df.sentence1_model != model) & (df.sentence2_model != model)
               cur_result['targeting']='untargeted'
          else:
               raise ValueError

          # split according to pair type (N vs S, S vs S)
          if pair_type is None:
               pass
          elif pair_type == 'N_vs_S':
               mask = mask & (df['pair_type']=='N_vs_S')
               cur_result['pair_type']='N_vs_S'
          elif pair_type == 'S_vs_S':
               mask = mask & (df['pair_type']=='S_vs_S')
               cur_result['pair_type']='S_vs_S'
          else:
               raise ValueError

          assert np.sum(mask)>0
          reduced_df = df[mask]

          # calculate within-subject correlations
          for subject in reduced_df['Participant Private ID'].unique():
               cur_subject_results=cur_result.copy()
               cur_subject_results['Participant Private ID']=subject
               mask=reduced_df['Participant Private ID']==subject
               cur_subject_reduced_df=reduced_df[mask]
               cur_subject_results['corr']=correlation_func(cur_subject_reduced_df['rating'],cur_subject_reduced_df[model])
               cur_subject_results['mean_rating_NC_corr']=correlation_func(cur_subject_reduced_df['rating'],cur_subject_reduced_df['mean_rating_NC'])
               df2.append(cur_subject_results)
     df2=pd.DataFrame(df2)
     return df2


def plot_four_conditions_corr(score_df,models,correlation_func='pearsonr', NC_measure='mean_rating_NC', chance_level=0.0,ylim=None):

     fig=plt.figure(figsize=(12,6.5))

     targeting = ['targeted','targeted','untargeted','untargeted']
     pair_type = ['N_vs_S','S_vs_S','N_vs_S','S_vs_S']
     title = ['N_vs_S, targeted','S_vs_S, targeted', 'N_vs_S, untargeted', 'S_vs_S, untargeted']

     for i_subplot in range(4):
          ax=plt.subplot(2,2,1+i_subplot)
          corr_df = get_correlation_based_accuracy(score_df, models,  correlation_func=correlation_func, targeting=targeting[i_subplot], pair_type=pair_type[i_subplot])

          mean_corr_df = corr_df.groupby('model').mean().drop(columns=['Participant Private ID'])
          mean_corr_df = mean_corr_df.rename(columns={'corr':'mean','mean_rating_NC_corr':'mean_rating_NC'})
          mean_corr_df['se'] = corr_df.groupby('model').std()['corr']/np.sqrt(corr_df.groupby('model').count()['corr'])
          mean_corr_df=mean_corr_df.reset_index()

          mean_corr_df['model'] = mean_corr_df['model'].astype("category")
          mean_corr_df.model.cat.set_categories(models, inplace=True)
          mean_corr_df=mean_corr_df.sort_values('model')

          print(mean_corr_df)
          plot_bars(ax, mean_corr_df, chance_level=chance_level, NC_measure=NC_measure, ylim=None)

          plt.title(title[i_subplot])

     plt.tight_layout()
     plt.show()

def model_by_model_heatmap(df, models=None, save_folder=None):
     if models is None:
          models = get_models(df)
     n_models = len(models)

     heatmap = np.empty((n_models,n_models))
     heatmap[:] = np.nan

     for i_row, synthetic_sentence_preferring_model in enumerate(models):
          for j_row, natural_sentence_preferring_model in enumerate(models):
               if natural_sentence_preferring_model == synthetic_sentence_preferring_model:
                    continue

               # filter trials
               df2 = filter_trials(df, targeted_model = synthetic_sentence_preferring_model,targeting='accept',trial_type='natural_vs_synthetic')
               df3 = filter_trials(df2, targeted_model = natural_sentence_preferring_model, targeting='reject',trial_type='natural_vs_synthetic')

               df3['subject_preferred_natural']=(
                                                  (df3['sentence1_type']=='N') & (df3['rating']<=3) |
                                                  (df3['sentence2_type']=='N') & (df3['rating']>=4)
                                                )

               # average human judgements
               proportion_natural_preferred = df3['subject_preferred_natural'].mean()
               heatmap[i_row,j_row] = proportion_natural_preferred

     axes_label_fontsize=10

     # plot heatmap

     matplotlib.rcParams.update({'font.size': 10})
     matplotlib.rcParams.update({'font.family':'sans-serif'})
     matplotlib.rcParams.update({'font.sans-serif':'Arial'})

     mask = np.eye(n_models, n_models,dtype=bool)

     widths_in_inches = [0.9,1.8,0.35]
     horizontal_elements=['left_margin','heatmaps','right_margin']
     heights_in_inches= [0.8,1.8,0.1,0.15,0.6]
     vertical_elements = ['top_margin','heatmaps','middle_margin','colorbar','bottom_margin']

     fig_w = np.sum(widths_in_inches)
     fig_h = np.sum(heights_in_inches)
     fig = plt.figure(figsize=(fig_w,fig_h))
     fig.set_size_inches(fig_w, fig_h)

     gs0=GridSpec(ncols=len(widths_in_inches), nrows=len(heights_in_inches), figure=fig, width_ratios=widths_in_inches,height_ratios=heights_in_inches, hspace=0, wspace=0,top=1,bottom=0,left=0,right=1)

     heatmap_ax = fig.add_subplot(gs0[vertical_elements.index('heatmaps'),horizontal_elements.index('heatmaps')])
     cbar_ax = fig.add_subplot(gs0[vertical_elements.index('colorbar'),horizontal_elements.index('heatmaps')])

     sns.heatmap(heatmap, mask=mask, xticklabels=niceify(models), yticklabels=niceify(models),
                 annot=True, fmt='.2f', cmap='bwr', vmin=0, vmax=1, center=0.5, square=True, linewidth=1.0,
                 ax = heatmap_ax, cbar_ax=cbar_ax, cbar_kws={'orientation':'horizontal'},
                 annot_kws={'fontsize':6})
     heatmap_ax.xaxis.set_ticks_position('top')
     heatmap_ax.tick_params('x', labelrotation=90)
     heatmap_ax.tick_params(axis='both', which='major', labelsize=8)
     heatmap_ax.tick_params(axis='both', which='minor', labelsize=8)
     cbar_ax.tick_params(axis='both', which='major', labelsize=8)
     cbar_ax.tick_params(axis='both', which='minor', labelsize=8)
     cbar_ax.set_xticks([0,0.25,0.5,0.75,1.0])
     cbar_ax.set_xlabel('humans preference of natural sentences\n(proportion of trials)',fontdict={'fontsize':axes_label_fontsize})
     heatmap_ax.set_ylabel('models assigned as $m_{accept}$',fontdict={'fontsize':axes_label_fontsize})
     heatmap_ax.set_xlabel('models assigned as $m_{reject}$',fontdict={'fontsize':axes_label_fontsize})
     heatmap_ax.xaxis.set_label_position('top')

     print(f"figure size: {fig_w},{fig_h} inches")

     if save_folder is not None:
               pathlib.Path(save_folder).mkdir(parents=True,exist_ok=True)
               fig.savefig(os.path.join(save_folder,'natural_vs_synthetic_human_preference_matrix.pdf'), dpi=600)
     else:
          plt.show()



if __name__ == '__main__':
# %% data preprocessing
     results_csv = 'behavioral_results/contstim_Aug2021_n100_results.csv'

     aligned_results_csv = results_csv.replace('.csv','_aligned.csv')
     aligned_results_csv_with_loso = results_csv.replace('.csv','_aligned_with_loso.csv')
     try:
          df=pd.read_csv(aligned_results_csv_with_loso)
     except:
          df=pd.read_csv(results_csv)
          df=align_sentences(df)
          df['sentence_pair']=[s1 + '_' + s2 for s1, s2 in zip(df['sentence1'],df['sentence2'])]
          df=recode_model_targeting(df)

          # %% remove excluded subjects
          #excluded_subjects=[3572610,3572431] 'behavioral_results/contstim_N32_results.csv'
          excluded_subjects=[]
          df=df[~df['Participant Private ID'].isin(excluded_subjects)]

          # there's one subject with two extra trials. eliminate them.
          # eliminate the repeated trials
          df=df.groupby(['Participant Private ID','sentence_pair']).first().reset_index()

          # anonymize subject IDs
          IDs, df['subject'] = np.unique(df['Participant Private ID'], return_inverse=True)
          df=df.drop(columns=['Participant Private ID'])

          # write down subject groups (each 10 subject group had the same trials)
          df['subject_group'] = [int(re.findall('set (\d+)_.',s)[0]) for s in df['counterbalance-o1ql']]

          df.to_csv(aligned_results_csv)
          pd.DataFrame(IDs).to_csv(aligned_results_csv.replace('.csv','_subject_ID_list.csv'))

          # add leave-one-out noise celing estimates
          df=add_leave_one_subject_predictions(df)
          df.to_csv(aligned_results_csv_with_loso)

# %% Binarized accuracy measurements
     # uncomment this next line to generate html result tables
     # build_all_html_files(df)

     # uncomment to plot main result figures
     # figs=plot_main_results_figures(df, save_folder = 'figures/binarized_acc')
     # plt.show()

     model_by_model_heatmap(df,save_folder='figures/heatmaps')


          # df_acc=get_binarized_accuracy(df,models)

          # print ('all trials:')
          # print(df_acc[models].mean(axis = 0, skipna = True))

          # # plot average accuracy across all trials
          # plt.figure()
          # ax=plt.gca()
          # models = get_models(df)
          # plot_bars(ax,average_models_within_conditions(df_acc, models),NC_measure='majority_vote_NC', chance_level=0.5, ylim=(0,1))

     #      # plot average accuracy within each condition
     #      plot_four_conditions(df_acc, models, NC_measure='majority_vote_NC',chance_level=0.5,ylim=(0,1))

     # # %% Correlation-based accuracy measurements

     #      score_df = log_prob_pairs_to_scores(df, transformation_func='diff')
     #      models = get_models(df)
     #      corr_df=get_correlation_based_accuracy(score_df,models, correlation_func='pearsonr')
     #      print(corr_df)
     #      plot_four_conditions_corr(score_df,models,correlation_func='spearmanr')



     # # print('predicting ratings from log-prob differences (pearson)')
     # # print(corr_df[models].mean(axis = 0, skipna = True))

     # # corr_df=get_correlation_based_accuracy(score_df,models,  correlation_func='pearsonr')
     # # print('predicting ratings from log-prob differences (pearsonr)')
     # # print(corr_df[models].mean(axis = 0, skipna = True))


     # # corr_df=get_correlation_based_accuracy(score_df,models,  correlation_func='spearmanr')
     # # print('predicting ratings from log-prob differences (spearman)')
     # # print(corr_df[models].mean(axis = 0, skipna = True))

     # # corr_df=get_correlation_based_accuracy(score_df,models, correlation_func='kendall-Tau-b')
     # # print('predicting ratings from log-prob differences (kendall-Tau-b)')
     # # print(corr_df[models].mean(axis = 0, skipna = True))


          # add a sentence pair field. This assumes that s1 and s2 are already sorted.

          #pairwise_binary_choice_analysis(df)