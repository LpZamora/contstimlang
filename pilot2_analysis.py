# %%
from collections import OrderedDict
import random
import re
import itertools
import math
import copy
import os, pathlib
import string
import inspect
from contextlib import contextmanager
import textwrap

import numpy as np
import pandas as pd
from pandas.core.reshape.reshape import get_dummies
from tqdm import tqdm
import pandas as pd
import scipy.stats
import statsmodels.stats.multitest
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as PathEffects
import matplotlib.patches as mpatches
from metroplot import metroplot

from signed_rank_cosine_similarity import calc_signed_rank_cosine_similarity_analytical_RAE, calc_expected_normalized_RAE_signed_rank_response_pattern, calc_semi_signed_rank_cosine_similarity_analytical_RAE

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

panel_letter_fontsize=12


# colors for scatter plot:
# https://www.visualisingdata.com/2019/08/five-ways-to-design-for-red-green-colour-blindness/
natural_sentence_color   = '#FFFFFF'
synthetic_sentence_color = '#AAAAAA'
shuffled_sentence_color  = '#000000'
selected_trial_color   =   '#000000'
unselected_trial_color =   '#AAAAAA'


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
               df.loc[idx_trial,'trial_type']='natural_vs_shuffled' # C1-> N (natural), C2-> C (catch)
               df.loc[idx_trial,'sentence1_type']=trial.sentence1_type.replace('C1','N').replace('C2','C')
               df.loc[idx_trial,'sentence2_type']=trial.sentence2_type.replace('C1','N').replace('C2','C')
          elif {trial.sentence1_type, trial.sentence2_type} == {'R1','R2'}:
               # randomly sampled natural sentences
               df.loc[idx_trial,'trial_type']='randomly_sampled_natural'
          else:
               raise ValueError

          # remove 1 and 2 from sentence type
          df.loc[idx_trial,'sentence1_type']=df.loc[idx_trial,'sentence1_type'].replace('1','').replace('2','')
          df.loc[idx_trial,'sentence2_type']=df.loc[idx_trial,'sentence2_type'].replace('1','').replace('2','')
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

def calc_binarized_accuracy(df, drop_model_prob = True):
     """ binarizes model and human predictions and returns 1 or 0 for prediction correctness """

     df2=df.copy()
     models = get_models(df)
     for model in models:

          assert not (df['sentence2_'+model + '_prob']==df['sentence1_'+model + '_prob']).any(), f'found tied prediction for model {model}'
          model_predicts_sent2=df['sentence2_'+model + '_prob']>df['sentence1_'+model + '_prob']
          human_chose_sent2=df['rating']>=4

          # store trial-level accuracy
          df2[model]=(model_predicts_sent2==human_chose_sent2).astype('float')

          # drop probability
          if drop_model_prob:
               df2=df2.drop(columns=['sentence1_'+model + '_prob','sentence2_'+model + '_prob'])

     df2['NC_LB']=(df2['majority_vote_NC_LB']==human_chose_sent2).astype(float)
     df2['NC_UB']=(df2['majority_vote_NC_UB']==human_chose_sent2).astype(float)
     return df2

def get_normalized_mean_RAE_signed_ranked_response(df, subject_group, excluded_subject=None):
     """ for a given subject group, calculate the expected normalized RAE signed-rank responses for each subject,
         and then average across subjects.

         the response pattern is returned as a new normalized_expected_RAE_signed_rank_response_pattern

         (used for noise ceiling bounds)
     """

     df_subject_group = df[df['subject_group']==subject_group]
     assert len(df_subject_group)>0

     if excluded_subject is not None:
          df_subject_group = df_subject_group[df_subject_group['subject']!=excluded_subject]

     if not 'zero_centered_rating' in df_subject_group.columns:
          df_subject_group['zero_centered_rating']=df_subject_group['rating']-3.5

     def add_expected_normalized_RAE_signed_rank_response_pattern(df):
          x = df['zero_centered_rating']
          r = calc_expected_normalized_RAE_signed_rank_response_pattern(x)
          df2=df.copy()
          df2['normalized_expected_RAE_signed_rank_response_pattern']=r
          return df2

     df_subject_group = df_subject_group.groupby('subject').apply(add_expected_normalized_RAE_signed_rank_response_pattern)

     # now reduce subjects
     df_subject_group = df_subject_group.groupby(['sentence_pair'],dropna=True).mean()
     df_subject_group = df_subject_group.drop(
          columns=set(df_subject_group.columns)-{'normalized_expected_RAE_signed_rank_response_pattern'})
     return df_subject_group

def RAE_signed_rank_cosine_similarity(df):
     """ ordinal correlation between log (p(s1|m)/p(s2|m)) and human ranks """

     df = df.copy()
     df['zero_centered_rating']=df['rating']-3.5

     models = get_models(df)

     subjects =  df['subject'].unique()
     results=[]

     for subject in tqdm(subjects):
          df_subject = df[df['subject']==subject]
          subject_group = df_subject['subject_group'].unique().item()

          cur_result={}
          cur_result['subject']=subject
          cur_result['subject_group']=subject_group

          # calculate human-model correlations
          for model in models:
               model_log_prob_diff = df_subject['sentence2_'+model + '_prob'] - df_subject['sentence1_'+model + '_prob']
               cur_result[model]=calc_signed_rank_cosine_similarity_analytical_RAE(model_log_prob_diff,df_subject['zero_centered_rating'])

          # and now for the noise ceiling bounds:
          def calculate_lower_bound_on_RAE_signed_rank_cosine_similarity_noise_ceiling(df, df_subject, subject, subject_group):
               df_subject = pd.concat(
                    [df_subject.set_index('sentence_pair'),
                    get_normalized_mean_RAE_signed_ranked_response(df, subject_group,excluded_subject=subject)],
                    axis=1)
               return calc_signed_rank_cosine_similarity_analytical_RAE(
                    df_subject['zero_centered_rating'],
                    df_subject['normalized_expected_RAE_signed_rank_response_pattern'])
          cur_result['NC_LB'] = calculate_lower_bound_on_RAE_signed_rank_cosine_similarity_noise_ceiling(df, df_subject, subject, subject_group)

          def calculate_upper_bound_on_RAE_signed_rank_cosine_similarity_noise_ceiling(df, df_subject, subject_group):
               df_subject = pd.concat(
                    [df_subject.set_index('sentence_pair'),
                    get_normalized_mean_RAE_signed_ranked_response(df, subject_group)],
                    axis=1)
               return calc_semi_signed_rank_cosine_similarity_analytical_RAE(
                    df_subject['zero_centered_rating'],
                    df_subject['normalized_expected_RAE_signed_rank_response_pattern']
               )
          cur_result['NC_UB'] = calculate_upper_bound_on_RAE_signed_rank_cosine_similarity_noise_ceiling(df, df_subject, subject_group)

          results.append(cur_result)
     return pd.DataFrame(results)

def calc_somersD(df):
     """ a simple ordinal correlation between log p(s1|m) - log p(s2|m) and human ranks
     """

     def _calc_somersD(model_log_prob_diff,human_rating):
          """ Calculate Somers' D ordinal correlation, assuming the model-to-human-rating mapping is symmetric for s1 vs. s2 and s2 vs. s1 trials """
          assert len(model_log_prob_diff) == len(human_rating)
          # negative_mask = model_log_prob_diff<0
          # model_log_prob_diff=np.where(negative_mask,-model_log_prob_diff,model_log_prob_diff)
          # human_rating = np.where(negative_mask, 7-human_rating,human_rating) # rating mapping: 1-->6, 2-->3, 3-->4, ..., 6-->1

          # model_log_prob_diff = np.concatenate([model_log_prob_diff,-model_log_prob_diff],axis=0)
          # human_rating = np.concatenate([human_rating,7-human_rating],axis=0)

          somersd_result = scipy.stats.somersd(y=model_log_prob_diff,x=human_rating)
          ordinal_corr = somersd_result.statistic
          return ordinal_corr

     models = get_models(df)
     subjects =  df['subject'].unique()

     results=[]

     for subject in subjects:
          df_subject = df[df['subject']==subject]
          subject_rating = df_subject['rating']

          cur_result={}
          cur_result['subject']=subject
          cur_result['subject_group']=df_subject['subject_group'].unique().item()

          for model in models:
               model_log_prob_diff = df_subject['sentence2_'+model + '_prob'] - df_subject['sentence1_'+model + '_prob']
               cur_result[model]=_calc_somersD(model_log_prob_diff,subject_rating)

          # now for the noise ceiling:
          # lower bound - use the other nine subjects' mean response
          leave_1_subject_out = df_subject['mean_rating_NC_LB']-3.5

          cur_result['NC_LB'] = _calc_somersD(leave_1_subject_out,subject_rating)
          cur_result['NC_UB'] = 1.0 # using non-informative bound, for now.

          results.append(cur_result)

     return pd.DataFrame(results)


def calc_flexible_somersD(df):
     """ a simple ordinal correlation between log p(s1|m) - log p(s2|m) and human ranks
     """

     def _calc_flexible_somersD(log_prob_s1_m,log_prob_s2_m,human_rating,leave_1_subject_out):
          """ Calculate Somers' D ordinal correlation, assuming the model-to-human-rating mapping is symmetric for s1 vs. s2 and s2 vs. s1 trials """
          assert len(log_prob_s1_m) == len(human_rating)
          # negative_mask = log_prob_s2_m<log_prob_s1_m
          # log_prob_s1_m_=np.where(negative_mask,log_prob_s2_m,log_prob_s1_m)
          # log_prob_s2_m_=np.where(negative_mask,log_prob_s1_m,log_prob_s2_m)
          # human_rating = np.where(negative_mask, 7-human_rating,human_rating) # rating mapping: 1-->6, 2-->3, 3-->4, ..., 6-->1

          log_prob_s1_m_ = np.concatenate([log_prob_s1_m,log_prob_s2_m],axis=0)
          log_prob_s2_m_ = np.concatenate([log_prob_s2_m,log_prob_s1_m],axis=0)
          log_prob_s1_m = log_prob_s1_m_
          log_prob_s2_m = log_prob_s2_m_
          human_rating = np.concatenate([human_rating,7-human_rating],axis=0)
          leave_1_subject_out = np.concatenate([leave_1_subject_out,7-leave_1_subject_out],axis=0)
          n_trials = len(log_prob_s1_m)

          # log_prob_s1_m=log_prob_s1_m.to_numpy()
          # log_prob_s2_m=log_prob_s2_m.to_numpy()
          # human_rating=human_rating.to_numpy()
          # leave_1_subject_out=leave_1_subject_out.to_numpy()

          n_total=0
          n_incongruent_LB_NC=0
          n_incongruent=0
          for i_trial in range(n_trials):
               for j_trial in range(n_trials):
                    if i_trial == j_trial:
                         continue
                    if (log_prob_s2_m[j_trial] >= log_prob_s2_m[i_trial]) and ((log_prob_s1_m[j_trial] <= log_prob_s1_m[i_trial])):
                         n_total+=1

                         if human_rating[j_trial] < human_rating[i_trial]:
                              n_incongruent +=1

                         if (
                            ((leave_1_subject_out[i_trial] > leave_1_subject_out[j_trial]) and (human_rating[i_trial] < human_rating[j_trial])) or
                            ((leave_1_subject_out[i_trial] < leave_1_subject_out[j_trial]) and (human_rating[i_trial] > human_rating[j_trial]))
                            ):
                              n_incongruent_LB_NC+=1

          if n_total == 0:
               n_total = 1
          return 1 - n_incongruent/n_total, 1-n_incongruent_LB_NC/n_total

     models = get_models(df)
     subjects =  df['subject'].unique()

     results=[]

     for subject in subjects:
          df_subject = df[df['subject']==subject]
          subject_rating = df_subject['rating']

          cur_result={}
          cur_result['subject']=subject
          cur_result['subject_group']=df_subject['subject_group'].unique().item()

          for model in models:
               log_prob_s1_m = df_subject['sentence1_'+model + '_prob']
               log_prob_s2_m = df_subject['sentence2_'+model + '_prob']
               leave_1_subject_out = df_subject['mean_rating_NC_LB']
               cur_result[model], cur_result['NC_LB_' + model]=_calc_flexible_somersD(log_prob_s1_m,log_prob_s2_m,subject_rating, leave_1_subject_out)
               cur_result['NC_UB_' + model] = 1.0 # using non-informative bound, for now.

          results.append(cur_result)

     return pd.DataFrame(results)

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


def model_specific_performace_dot_plot(df, models, ylabel='% accuracy',title=None,ax=None, each_dot_is='subject_group',chance_level=None, model_specific_NC=False, pairwise_sig=None, tick_label_fontsize=8, measure='binarized_accuracy'):

     matplotlib.rcParams.update({'font.size': 10})
     matplotlib.rcParams.update({'font.family':'sans-serif'})
     matplotlib.rcParams.update({'font.sans-serif':'Arial'})

     if ax is None:
          plt.figure(figsize=(4.5,2.4))
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

     if measure=='binarized_accuracy':
          plt.xlim([0.0,1.0])
          plt.xticks([0,0.25,0.5,0.75,1.0],['0','25%','50%','75%','100%'])
     else:
          plt.xlim([-1.0,1.0])
          plt.xticks([-1,-0.5,0,0.5,1.0])
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

def plot_one_main_results_panel(df, reduction_fun, models, cur_panel_cfg, ax = None, metroplot_ax = None, chance_level=None, metroplot_preallocated_positions=None, tick_label_fontsize=8, measure='binarized_accuracy'):
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


     model_specific_NC=cur_panel_cfg['only_targeted_trials']
     if measure=='flexible_SomersD':
          model_specific_NC=True

     model_specific_performace_dot_plot(reduced_df,models, ylabel='% accuracy',title=None,ax=ax, each_dot_is='subject_group', chance_level=chance_level, model_specific_NC=model_specific_NC, pairwise_sig=pairwise_sig, tick_label_fontsize=tick_label_fontsize, measure=measure)

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

def catch_trial_report(df):
     """ generate statistics on subject performance in catch trials"""

     df2=df[df['trial_type']=='natural_vs_shuffled']

     # how many unique catch trials were presented?
     # distribution of errors

     n_catch_trials = int(df2.groupby('subject')['sentence_pair'].count().unique())
     print(f'each subject was presented with {n_catch_trials} catch trials.')

     df2['correct_catch']=((df2['sentence1_type']=='N') & (df2['rating']<=3)) | ((df2['sentence2_type']=='N') & (df2['rating']>=4))
     subject_specific_correct_catch_trials = df2.groupby('subject')['correct_catch'].sum('correct_catch')
     print('distribution of correct catch trials (subjects):\n',subject_specific_correct_catch_trials.value_counts())


def generate_worst_sentence_pairs_table(df, trial_type  = 'natural_controversial', targeting=None, models=None, n_sentences_per_model=1):
     if models is None:
          models = get_models(df)
     latex_table = pd.DataFrame()

     for min_acc_model in models:

          df_filtered = filter_trials(df, targeted_model = min_acc_model, trial_type=trial_type, targeting=targeting)

          acc_df = calc_binarized_accuracy(df_filtered, drop_model_prob=False).groupby('sentence_pair').mean(numeric_only=True)

          sorted_acc_df=acc_df.sort_values(by=min_acc_model,axis='index',ascending=True,)

          def build_table_row(latex_table, cur_sentence_pair, model_to_show_as_model1):
               df_reduced=df[df['sentence_pair']==cur_sentence_pair]

               sentence1 = df_reduced.sentence1.unique()[0]
               sentence2 = df_reduced.sentence2.unique()[0]
               model1=df_reduced.sentence1_model.unique()[0]
               model2=df_reduced.sentence2_model.unique()[0]
               p_s1_m1 = df_reduced[f'sentence1_{model1}_prob'].mean()
               p_s1_m2 = df_reduced[f'sentence1_{model2}_prob'].mean()
               p_s2_m1 = df_reduced[f'sentence2_{model1}_prob'].mean()
               p_s2_m2 = df_reduced[f'sentence2_{model2}_prob'].mean()
               s1_human_choices = str(int((df_reduced.rating<=3).sum()))
               s2_human_choices = str(int((df_reduced.rating>=4).sum()))
               s1_type = df_reduced.sentence1_type.unique()
               s2_type = df_reduced.sentence2_type.unique()

               if model1 != model_to_show_as_model1:
                    model1, model2, p_s1_m1, p_s1_m2, p_s2_m1, p_s2_m2 = model2, model1, p_s1_m2, p_s1_m1, p_s2_m2, p_s2_m1

               # flip sentences so the first sentence is always the one the humans chose
               if s2_human_choices > s1_human_choices:
                    sentence1, sentence2, s1_human_choices, s2_human_choices = sentence2, sentence1, s2_human_choices, s1_human_choices
                    p_s1_m1, p_s1_m2, p_s2_m1, p_s2_m2 = p_s2_m1, p_s2_m2, p_s1_m1, p_s1_m2
                    s1_type, s2_type = s2_type, s1_type

               if trial_type == 'natural_controversial':
                    s1_notation, s2_notation =  'n_1', 'n_2'
               elif trial_type == 'synthetic_vs_synthetic':
                    s1_notation, s2_notation =  's_1', 's_2'
               elif trial_type == 'natural_vs_synthetic':
                    # make sure the natural sentence is presented as the first sentence
                    if (s1_type == 'S') and (s2_type == 'N'):
                         s1_notation, s2_notation =  's', 'n'
                    else:
                         assert (s1_type== 'N') and (s2_type == 'S')
                         s1_notation, s2_notation =  'n', 's'

               # sentence 1
               cur_row = dict()
               cur_row['sentence']=f'${s1_notation}$: {sentence1}.'
               cur_row['log probability (model 1)']=f'$\log p({s1_notation} | \\textrm{{{niceify(model1)}}})=$\\num{{{p_s1_m1:.2f}}}'
               cur_row['log probability (model 2)']=f'$\log p({s1_notation} | \\textrm{{{niceify(model2)}}})=$\\num{{{p_s1_m2:.2f}}}'
               cur_row['\# human choices'] = f'\\num{{{s1_human_choices}}}'

               def make_bold_num(s):
                    return s.replace(r'\num{',r'\textbf{\num{') + r'}'

               if p_s1_m1 > p_s2_m1: # make numbers bold
                    cur_row['log probability (model 1)']=make_bold_num(cur_row['log probability (model 1)'])
               if p_s1_m2 > p_s2_m2:
                    cur_row['log probability (model 2)']=make_bold_num(cur_row['log probability (model 2)'])
               if s1_human_choices > s2_human_choices:
                    cur_row['\# human choices']=make_bold_num(cur_row['\# human choices'])
               latex_table = latex_table.append(cur_row, ignore_index=True)

               # sentence 2
               cur_row = dict()
               cur_row['sentence']=f'${s2_notation}$: {sentence2}.'
               cur_row['log probability (model 1)']=f'$\log p({s2_notation} | \\textrm{{{niceify(model1)}}})=$\\num{{{p_s2_m1:.2f}}}'
               if p_s2_m1 > p_s1_m1: # make numbers bold
                    cur_row['log probability (model 1)']=make_bold_num(cur_row['log probability (model 1)'])
               cur_row['log probability (model 2)']=f'$\log p({s2_notation} | \\textrm{{{niceify(model2)}}})=$\\num{{{p_s2_m2:.2f}}}'
               if p_s2_m2 > p_s1_m2: # make numbers bold
                    cur_row['log probability (model 2)']=make_bold_num(cur_row['log probability (model 2)'])
               cur_row['\# human choices'] = f'\\num{{{s2_human_choices}}}'
               # if s2_human_choices > s1_human_choices:
               #      cur_row['\# human choices']=make_bold_num(cur_row['\# human choices'])
               latex_table = latex_table.append(cur_row, ignore_index=True)

               latex_table=latex_table[['sentence','log probability (model 1)','log probability (model 2)','\# human choices']]
               return latex_table


          for i_sentence in range(n_sentences_per_model):
               # take one sentence of minimal accuracy
               worst_acc = sorted_acc_df[min_acc_model].min()
               cur_candidate_sentence_pairs = sorted_acc_df[sorted_acc_df[min_acc_model]==worst_acc]
               print(min_acc_model,len(cur_candidate_sentence_pairs))
               cur_sentence_pair = cur_candidate_sentence_pairs.sample(n=1).iloc[0].name
               sorted_acc_df = sorted_acc_df.drop(cur_sentence_pair)

               # add it to the table
               latex_table = build_table_row(latex_table, cur_sentence_pair, model_to_show_as_model1 = min_acc_model)
     with pd.option_context("max_colwidth", 1000): # https://stackoverflow.com/a/67419969
          print(latex_table)
          latex_code = latex_table.to_latex(header=True, index=False, escape=False)

     # some tweaks:

     # tabularx
     latex_code = latex_code.replace(r'\begin{tabular}{llll}',r'\begin{tabularx}{\textwidth}{lllc}')
     latex_code = latex_code.replace(r'\end{tabular}',r'\end{tabularx}')

     # align s and n sentences (this applies to s vs n trials)
     if trial_type == 'natural_vs_synthetic':
          latex_code = latex_code.replace(r'$s$: ',r'\makebox[0pt][l]{$s$: }\hphantom{$n$: }')
          latex_code = latex_code.replace(r'p(s |',r'p(\mathmakebox[0pt][l]{s}\hphantom{n} |')

     # add midrules to group sentence pairs
     latex_lines = latex_code.split(r'\\')
     for i in range(len(latex_lines)):
          if i % 2 and i > 1 and i<(len(latex_lines)-1):
               latex_lines[i] = r'\midrule'+ latex_lines[i]
     latex_code = r'\\'.join(latex_lines)

     pathlib.Path('tables').mkdir(parents=True,exist_ok=True)
     tex_fname = f'tables/{trial_type}.tex'
     if targeting is not None:
          tex_fname = tex_fname.replace('.tex','_'+ targeting + '.tex')
     with open(tex_fname, "w") as tex_file:
          tex_file.write(latex_code)

def plot_main_results_figures(df, models=None, plot_metroplot=True, save_folder = None, measure='binarized_accuracy', initial_panel_letter_index=0):
     if models is None:
          models = get_models(df)

     if measure == 'binarized_accuracy':
          reduction_fun = calc_binarized_accuracy
          chance_level=0.5
     elif measure == 'SomersD':
          reduction_fun = calc_somersD
          chance_level=0
     elif measure == 'Pearsons_r':
          reduction_fun = calc_r
          chance_level=0
     elif measure == 'flexible_SomersD':
          chance_level=0
          reduction_fun=calc_flexible_somersD
     elif measure == 'RAE_signed_rank_cosine_similarity':
          chance_level=0
          reduction_fun=RAE_signed_rank_cosine_similarity
     else:
          raise ValueError

     # define figure structure
     panel_cfg  = [
          {'title':'Randomly sampled natural-sentence pairs',    'only_targeted_trials':False,     'trial_type':'randomly_sampled_natural',    'targeting':None,},
          {'title':'Controversial natural-sentence pairs',       'only_targeted_trials':True,      'trial_type':'natural_controversial',       'targeting':None,},
          {'title':'Synthetic controversial sentence pairs',     'only_targeted_trials':True,      'trial_type':'synthetic_vs_synthetic',      'targeting':None,},
          {'title':'Synthetic vs. natural sentences',            'only_targeted_trials':True,      'trial_type':'natural_vs_synthetic',        'targeting':'accept',},
          {'title':None,                                         'only_targeted_trials':False,     'trial_type':None,                          'targeting':None,},
          {'title':None,                                         'only_targeted_trials':True,      'trial_type':'natural_vs_synthetic',        'targeting':'reject'},
     ]

     figure_plans = [
          {'panels':[0,1], 'fname':f'natural_and_natural_controversial_{measure}.pdf', 'include_scatter_plot_col':True, 'include_panel_letters':True},
          {'panels':[2,3], 'fname':f'synthetic_{measure}.pdf', 'include_scatter_plot_col':True, 'include_panel_letters':True, 'include_scatter_plot_legend':True},
          {'panels':[4],   'fname':f'all_trials_{measure}.pdf', 'include_scatter_plot_col':False, 'include_panel_letters':False},
          {'panels':[5], 'fname':f'synthetic_vs_natural_reject_{measure}.pdf', 'include_scatter_plot_col':False, 'include_panel_letters':False},
     ]

     dotplot_xaxis_label = {
          'binarized_accuracy':'human-choice prediction accuracy',
          'SomersD':"ordinal correlation between human ratings and models'\nsentence pair probability log-ratio (Somers' D)",
          'flexible_SomersD':"modified Somers' D between human ratings and models'\nsentence probabilities",
          'RAE_signed_rank_cosine_similarity':"ordinal correlation between human ratings and models'\nsentence pair probability log-ratio (signed-rank cosine similarity)"
     }[measure]

     # all of the following measures are in inches
     left_margin = 0.6 # space to the left of the scatter plot
     right_margin = 0.00
     panel_h = 1.8
     panel_w = 1.8

     golden = (1 + 5 ** 0.5) / 2
     scatter_plot_w = 1.8/golden # space reserved for scatter plot
     scatter_plot_h = 1.8/golden # space reserved for scatter plot (1.2 for smaller size)

     v_space_above_panel = 0.2 # 0.25 for larger panels, 0.2 for smaller panels

     if '\n' in dotplot_xaxis_label: # allocate more space for two-line x-axis label
          v_space_below_panel = 0.535
     else:
          v_space_below_panel = 0.4

     # v_space_below_panel = v_space_below_panel + 9/72 # for bigger size scatter plot

     v_space_between_rows = 6/72

     h_space1 = 0.85 # horizontal space between scatter plot and result panel
     left_margin_narrow = 1.1 # left margin used for single column figures
     h_space2 = 0.05 # horizontal space between result column and metroplot
     metroplot_w = 1.2
     metroplot_preallocated_positions=8 # how many significance elements are we expecting.

     title_vertical_shift = 0 #16/72
     title_horizontal_shift = 12/72

     dotplot_panel_letter_horizontal_shift = 0.6  # distance between dotplot y-axis and panel letter
     dotplot_panel_letter_vertical_shift = 6/72# distance between dotplot top edge and panel letter (9/72 for larger, 6/72 for smaller)
     scatterplot_panel_letter_horizontal_shift = 0.54 # distance between scatterplot y-axis and panel letter
     panel_title_fontsize = 10
     axes_label_fontsize = 10
     tick_label_fontsize = 8

     figs = []
     for i_figure, figure_plan in enumerate(figure_plans):

          # setup gridspec grid structure
          n_panels = len(figure_plan['panels'])

          if isinstance(initial_panel_letter_index,list):
               panel_letter_index=initial_panel_letter_index[i_figure] # panel counter (0 for a, 1 for b, and so on)
          else:
               panel_letter_index=initial_panel_letter_index

          # vertical structure
          panel_height = v_space_above_panel + panel_h + v_space_below_panel
          fig_h = panel_height*n_panels + v_space_between_rows*(n_panels-1) # height is set adaptively
          fig_hspace = v_space_between_rows/panel_height

          # determine horizontal structure
          if figure_plan['include_scatter_plot_col']: # scatter plot on left, results dot plot on right
               widths_in_inches = [left_margin, scatter_plot_w, h_space1, panel_w, h_space2, metroplot_w, right_margin]
               horizontal_elements = [None,'scatter_plot',None,'dot_plot',None,'metroplot',None]
          else: # only a single coloumn
               widths_in_inches = [left_margin_narrow, panel_w, h_space2, metroplot_w, right_margin]
               horizontal_elements = [None,'dot_plot',None,'metroplot',None]
          fig_w = np.sum(widths_in_inches)

          print(f"figure {figure_plan['fname']} size: {fig_w},{fig_h} inches")
          fig = plt.figure(figsize=(fig_w,fig_h))

          gs0=GridSpec(ncols=1, nrows=n_panels, figure=fig,hspace=fig_hspace, wspace=0,
          top=1,bottom=0,
          left=0,right=1)

          gs_row_list = [] # this list contains the grispecs for panel-specific row gridspecs
          gs_inner_list = []

          for i_panel, panel_idx in enumerate(figure_plan['panels']):
               cur_panel_cfg = panel_cfg[panel_idx]

               # build a vertical subgridspec within gs_row
               gs_row = gs0[i_panel].subgridspec(nrows=3,ncols=1,wspace=0,hspace=0,height_ratios=[v_space_above_panel,panel_h,v_space_below_panel])
               gs_row_list.append(gs_row)

               # build a horizontal subgridspec within the central cell of gs_row
               gs_inner = gs_row[1].subgridspec(nrows=1,ncols=len(horizontal_elements),wspace=0,hspace=0,width_ratios=widths_in_inches)
               gs_inner_list.append(gs_inner_list)

               result_panel_ax=fig.add_subplot(gs_inner[horizontal_elements.index(f'dot_plot')])

               if plot_metroplot:
                    metroplot_ax = fig.add_subplot(gs_inner[horizontal_elements.index(f'metroplot')])
               else:
                    metroplot_ax = None

               panel_top_edge = (i_panel*(panel_height+v_space_between_rows)+v_space_above_panel)-dotplot_panel_letter_vertical_shift
               panel_top_edge = 1 - panel_top_edge/fig_h # figure relative coordinates
               def get_panel_left_edge(which_panel='dot_plot'):
                    if which_panel == 'dot_plot':
                         panel_left_edge =  (np.cumsum(widths_in_inches)[horizontal_elements.index('dot_plot')-1]-dotplot_panel_letter_horizontal_shift)
                    elif which_panel == 'scatter_plot':
                         panel_left_edge =  (np.cumsum(widths_in_inches)[horizontal_elements.index('scatter_plot')-1]-scatterplot_panel_letter_horizontal_shift)
                    panel_left_edge = panel_left_edge / fig_w # figure relative coordinates
                    return panel_left_edge

               # plot scatter plots?
               if figure_plan['include_scatter_plot_col']:

                    x_model='gpt2'
                    y_model='roberta'

                    # Since the scatter plot and the dot plot reside within the same container and the scatter plot is smaller,
                    # we'd like to pad the scatter plot from above and below so it is vertically centered.

                    scatter_gs = gs_inner[horizontal_elements.index(f'scatter_plot')].subgridspec(nrows=3,ncols=1,wspace=0,hspace=0,height_ratios=[(panel_w-scatter_plot_h)/2,scatter_plot_h,(panel_w-scatter_plot_h)/2])
                    scatter_plot_ax = fig.add_subplot(scatter_gs[1])
                    if cur_panel_cfg['only_targeted_trials']:
                         targeted_model_1=x_model
                         targeted_model_2=y_model
                         if hasattr(cur_panel_cfg,'use_targeting_in_scatter_plot') and cur_panel_cfg['use_targeting_in_scatter_plot']:
                              if cur_panel_cfg['targeting']=='reject':
                                   targeting_1 = 'reject'
                                   targeting_2 = 'accept'
                              elif cur_panel_cfg['targeting']=='accept':
                                   targeting_1 = 'accept'
                                   targeting_2 = 'reject'
                              else:
                                   targeting_1 = None
                                   targeting_2 = None
                         else:
                              targeting_1 = None
                              targeting_2 = None
                    else:
                         targeting_1 = None
                         targeting_2 = None
                         targeted_model_1=None
                         targeted_model_2=None

                    sentence_pair_scatter_plot(df, x_model=x_model, y_model=y_model,
                                               trial_type=cur_panel_cfg['trial_type'], targeting_1=targeting_1, targeting_2=targeting_2,
                                               targeted_model_1=targeted_model_1,targeted_model_2=targeted_model_2,
                                               ax = scatter_plot_ax, axes_label_fontsize=tick_label_fontsize, tick_label_fontsize=tick_label_fontsize)
               if ('include_scatter_plot_legend' in figure_plan) and figure_plan['include_scatter_plot_legend']:
                    # scatter_plot_legend_ax = fig.add_subplot(scatter_gs[2])
                    scatter_plot_legend(fig, fig_w, fig_h)


               plot_one_main_results_panel(df, reduction_fun, models, cur_panel_cfg, ax=result_panel_ax, chance_level=chance_level, metroplot_ax=metroplot_ax, metroplot_preallocated_positions=metroplot_preallocated_positions, tick_label_fontsize=tick_label_fontsize, measure=measure)

               # add panel a letter to the left (only if no scatter plot)
               if figure_plan['include_scatter_plot_col']:
                    panel_letter_x = get_panel_left_edge(which_panel='scatter_plot')
               else:
                    panel_letter_x = get_panel_left_edge(which_panel='dot_plot')

               if figure_plan['include_panel_letters']:
                    plt.figtext(x=panel_letter_x,
                              y=panel_top_edge,
                              s=string.ascii_letters[panel_letter_index],
                              fontdict={'fontsize':panel_letter_fontsize,'weight':'bold'}
                              )
                    panel_letter_index+=1


               result_panel_ax.set_ylabel('')

               # if i_panel == n_panels-1:
               result_panel_ax.set_xlabel(dotplot_xaxis_label,fontdict={'fontsize':axes_label_fontsize})

               if cur_panel_cfg['title'] is not None:
                    # title_ax = fig.add_subplot(gs_row[0])
                    # title_ax.set_title(cur_panel_cfg['title'],y=0.1,fontdict={'fontsize':panel_title_fontsize})
                    # title_ax.set_axis_off()

                    title_left_edge = panel_letter_x + title_horizontal_shift/fig_w

                    plt.figtext(x=title_left_edge,
                           y=panel_top_edge+title_vertical_shift/fig_h,
                           s=cur_panel_cfg['title'],
                           fontdict={'fontsize':panel_title_fontsize}
                           )

          if save_folder is not None:
               pathlib.Path(save_folder).mkdir(parents=True,exist_ok=True)
               fig.savefig(os.path.join(save_folder,figure_plan['fname']), dpi=600)
               figs.append(fig)
     return figs

def _rank_transform_sentence_probs(df,model, percentile=True):
     new_df = pd.DataFrame({
          'sentence':pd.concat([df['sentence1'],  df['sentence2']]),
          'log_prob':pd.concat([df[f'sentence1_{model}_prob'],   df[f'sentence2_{model}_prob']])
          })
     new_df=new_df.drop_duplicates(subset='sentence')
     new_df['rank']=new_df.log_prob.rank(method='dense')
     new_df['rank']=new_df['rank']/new_df['rank'].max()
     if percentile:
          new_df['rank']=new_df['rank']*100

     df2 = df.copy()
     df2=df2.merge(new_df,left_on='sentence1',right_on='sentence',validate='many_to_one',how='left')
     df2=df2.drop(columns=['sentence','log_prob']).rename(columns={'rank':f'sentence1_{model}_rank'})
     df2=df2.merge(new_df,left_on='sentence2',right_on='sentence',validate='many_to_one',how='left')
     df2=df2.drop(columns=['sentence','log_prob']).rename(columns={'rank':f'sentence2_{model}_rank'})

     return df2

def sentence_pair_scatter_plot(df, x_model, y_model, trial_type=None, targeting_1=None, targeting_2=None, targeted_model_1=None, targeted_model_2=None,
                               ax=None,  axes_label_fontsize=10, tick_label_fontsize=8):

     # first, rank transform all sentences across the experiment

     df2 = df.copy()

     df2 = df2.drop_duplicates(subset='sentence_pair',keep='first')

     df2 = _rank_transform_sentence_probs(df2, x_model,percentile=True)
     df2 = _rank_transform_sentence_probs(df2, y_model,percentile=True)

     df2_filtered = filter_trials(df2, targeted_model = targeted_model_1,targeting=targeting_1,trial_type=trial_type)
     df2_filtered = filter_trials(df2_filtered, targeted_model = targeted_model_2,targeting=targeting_2,trial_type=trial_type)
     df2['trial_selected']=df2['sentence_pair'].isin(df2_filtered['sentence_pair'])


     color = []

     for idx, trial in df2.iterrows():
          for i_sentence in [1,2]:
               if trial['trial_selected']:
                    if trial[f'sentence{i_sentence}_type'] in ['R','N']:
                         df2.loc[idx,f'sentence{i_sentence}_color'] = natural_sentence_color
                    elif trial[f'sentence{i_sentence}_type'] in ['S']:
                         df2.loc[idx,f'sentence{i_sentence}_color'] = synthetic_sentence_color
                    elif trial[f'sentence{i_sentence}_type'] in ['C']:
                         df2.loc[idx,f'sentence{i_sentence}_color'] = shuffled_sentence_color
               else:
                    df2.loc[idx,f'sentence{i_sentence}_color']=unselected_trial_color


     df2_filtered = filter_trials(df2, targeted_model = targeted_model_1,targeting=targeting_1,trial_type=trial_type)
     df2_filtered = filter_trials(df2_filtered, targeted_model = targeted_model_2,targeting=targeting_2,trial_type=trial_type)

     if ax is None:
          fig = plt.figure()
          ax = plt.gca()

     ax.scatter(x=df2_filtered[f'sentence1_{x_model}_rank'],y=df2_filtered[f'sentence1_{y_model}_rank'],c=df2_filtered[f'sentence1_color'],s=10,zorder=100, edgecolors = 'k',linewidths=0.1, clip_on=False)
     ax.scatter(x=df2_filtered[f'sentence2_{x_model}_rank'],y=df2_filtered[f'sentence2_{y_model}_rank'],c=df2_filtered[f'sentence2_color'],s=10,zorder=100, edgecolors = 'k',linewidths=0.1, clip_on=False)
     for idx, trial in df2_filtered.iterrows():
          ax.plot([trial[f'sentence1_{x_model}_rank'],trial[f'sentence2_{x_model}_rank']],
                   [trial[f'sentence1_{y_model}_rank'],trial[f'sentence2_{y_model}_rank']],
              marker=None,
              color=selected_trial_color,
              linewidth=0.1, zorder=0,alpha=0.5)
     ax.set_xlim([0,100])
     ax.set_ylim([0,100])
     ax.set_aspect('equal','box')
     ax.set_xlabel(f'{niceify(x_model)}\np(sentence) percentile', fontsize=axes_label_fontsize)
     ax.set_ylabel(f'{niceify(y_model)}\np(sentence) percentile', fontsize=axes_label_fontsize)
     ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
     ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)

def scatter_plot_legend(fig, fig_w, fig_h):
     # https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
     legend_elements = [ # this styles should match the scatter plot in sentence_pair_scatter_plot
          Line2D([0], [0], marker='o', color='w', label='natural sentence',markerfacecolor=natural_sentence_color, markeredgecolor='k', markersize=np.sqrt(10), markeredgewidth=0.1),
          Line2D([0], [0], marker='o', color='w', label='synthetic sentence',markerfacecolor=synthetic_sentence_color, markeredgecolor='k', markersize=np.sqrt(10),  markeredgewidth=0.1),
     ]
     # explanation about sqrt marker size : https://stackoverflow.com/a/47403507/
     fig.legend(
          handles=legend_elements, loc=[(-6/72)/fig_w,(6/72)/fig_h], ncol=2,
          fontsize=8, handletextpad=0.0, labelspacing=0.2, frameon=False,
          columnspacing=0.5, borderaxespad=0,
          borderpad=0., handleheight=0.5)


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

def model_by_model_N_vs_S_heatmap(df, models=None, save_folder=None):
     if models is None:
          models = get_models(df)
     n_models = len(models)

     heatmap = np.empty((n_models,n_models))
     heatmap[:] = np.nan

     for i_row, m_accept in enumerate(models): # synthetic sentence preferring model
          for i_col, m_reject in enumerate(models): # natural sentence preferring model
               if m_reject == m_accept:
                    continue

               # filter trials
               df2 = filter_trials(df, targeted_model = m_accept,targeting='accept',trial_type='natural_vs_synthetic')
               df3 = filter_trials(df2, targeted_model = m_reject, targeting='reject',trial_type='natural_vs_synthetic')

               n_aligned_with_m_accept=0
               n_aligned_with_m_reject=0
               for i_trial, trial in df3.iterrows():
                    if ((trial[f'sentence1_{m_accept}_prob'] < trial[f'sentence2_{m_accept}_prob']) and
                        (trial[f'sentence1_{m_reject}_prob'] > trial[f'sentence2_{m_reject}_prob'])):
                        # m_accept preferred sentence 2, m_reject preferred sentence 1
                         if trial['rating']>=4: # subject preferred sentence 2 (aligned with m_accept)
                              n_aligned_with_m_accept+=1
                         elif trial['rating']<=3: # subject preferred sentence 1 (aligned with model2)
                              n_aligned_with_m_reject+=1
                    elif ((trial[f'sentence1_{m_accept}_prob'] > trial[f'sentence2_{m_accept}_prob']) and
                          (trial[f'sentence1_{m_reject}_prob'] < trial[f'sentence2_{m_reject}_prob'])):
                         # m_accept preferred sentence 1, m_reject preferred sentence 2
                         if trial['rating']>=4: # subject preferred sentence 2 (aligned with m_reject)
                              n_aligned_with_m_reject+=1
                         elif trial['rating']<=3: # subject preferred sentence 1 (aligned with m_accept)
                              n_aligned_with_m_accept+=1
                    else:
                         raise ValueError
               # average human judgements
               heatmap[i_row,i_col] = n_aligned_with_m_accept/(n_aligned_with_m_accept+n_aligned_with_m_reject)

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
                 annot=True, fmt='.2f', cmap='PiYG', vmin=0, vmax=1, center=0.5, square=True, linewidth=1.0,
                 ax = heatmap_ax, cbar_ax=cbar_ax, cbar_kws={'orientation':'horizontal','ticks':[0,0.25,0.5,0.75,1.0]},
                 annot_kws={'fontsize':6})
     heatmap_ax.xaxis.set_ticks_position('top')
     heatmap_ax.tick_params('x', labelrotation=90)
     heatmap_ax.tick_params(axis='both', which='major', labelsize=8)
     heatmap_ax.tick_params(axis='both', which='minor', labelsize=8)
     cbar_ax.tick_params(axis='both', which='major', labelsize=8)
     cbar_ax.tick_params(axis='both', which='minor', labelsize=8)
     cbar_ax.set_xlabel('humans choice aligned with $m_{accept}$\n(proportion of trials)',fontdict={'fontsize':axes_label_fontsize})
     heatmap_ax.set_ylabel('models assigned as $m_{accept}$',fontdict={'fontsize':axes_label_fontsize})
     heatmap_ax.set_xlabel('models assigned as $m_{reject}$',fontdict={'fontsize':axes_label_fontsize})
     heatmap_ax.xaxis.set_label_position('top')

     print(f"figure size: {fig_w},{fig_h} inches")

     if save_folder is not None:
               pathlib.Path(save_folder).mkdir(parents=True,exist_ok=True)
               fig.savefig(os.path.join(save_folder,'natural_vs_synthetic_human_preference_matrix.pdf'), dpi=600)
     else:
          plt.show()

def model_by_model_consistency_heatmap(df, models=None, save_folder=None, trial_type = None):
     if models is None:
          models = get_models(df)
     n_models = len(models)

     heatmap = np.empty((n_models,n_models))
     heatmap[:] = np.nan

     for i_row, model1 in enumerate(models):
          for i_col, model2 in enumerate(models):
               if model1 == model2:
                    continue

               # filter trials
               df2 = filter_trials(df, targeted_model = model1, trial_type=trial_type)
               df2 = filter_trials(df2, targeted_model = model2)

               n_aligned_with_model1=0
               n_aligned_with_model2=0
               for i_trial, trial in df2.iterrows():
                    if ((trial[f'sentence1_{model1}_prob'] < trial[f'sentence2_{model1}_prob']) and
                        (trial[f'sentence1_{model2}_prob'] > trial[f'sentence2_{model2}_prob'])):
                        # model1 preferred sentence 2, model2 preferred sentence 1
                         if trial['rating']>=4: # subject preferred sentence 2 (aligned with model1)
                              n_aligned_with_model1+=1
                         elif trial['rating']<=3: # subject preferred sentence 1 (aligned with model2)
                              n_aligned_with_model2+=1
                    elif ((trial[f'sentence1_{model1}_prob'] > trial[f'sentence2_{model1}_prob']) and
                          (trial[f'sentence1_{model2}_prob'] < trial[f'sentence2_{model2}_prob'])):
                         # model1 preferred sentence 1, model2 preferred sentence 2
                         if trial['rating']>=4: # subject preferred sentence 2 (aligned with model2)
                              n_aligned_with_model2+=1
                         elif trial['rating']<=3: # subject preferred sentence 1 (aligned with model1)
                              n_aligned_with_model1+=1

               # average human judgements
               heatmap[i_row,i_col] = n_aligned_with_model1/(n_aligned_with_model1+n_aligned_with_model2)

def model_by_model_agreement_heatmap(df, models=None, save_folder=None, trial_type = 'randomly_sampled_natural'):
     """ regardless of human choices, how often each pair of models agreed on which sentence is more probable."""

     if models is None:
          models = get_models(df)
     n_models = len(models)

     heatmap = np.empty((n_models,n_models))
     heatmap[:] = np.nan

     min_agreement = np.inf
     min_agreement_models = ()
     max_agreement = -np.inf
     max_agreement_models = ()

     for i_row, model1 in enumerate(models):
          for i_col, model2 in enumerate(models):
               if model1 == model2:
                    continue

               # filter trials
               df2 = filter_trials(df, trial_type=trial_type)

               n_congruent=0
               n_incongruent=0
               for i_trial, trial in df2.iterrows():
                    model1_sign = np.sign(trial[f'sentence1_{model1}_prob']-trial[f'sentence2_{model1}_prob'])
                    model2_sign = np.sign(trial[f'sentence1_{model2}_prob']-trial[f'sentence2_{model2}_prob'])
                    assert model1_sign!=0, 'this function assumes that there are no sentence probability ties'
                    assert model2_sign!=0, 'this function assumes that there are no sentence probability ties'
                    if model1_sign == -model2_sign:
                        n_incongruent+=1
                    elif model1_sign == model2_sign:
                        n_congruent+=1
               agreement_rate = n_congruent/(n_congruent+n_incongruent)
               heatmap[i_row,i_col] = agreement_rate

               if agreement_rate < min_agreement:
                    min_agreement = agreement_rate
                    min_agreement_models = model1, model2

               if agreement_rate > max_agreement:
                    max_agreement = agreement_rate
                    max_agreement_models = model1, model2

     print (f'min agreement: {min_agreement:.1%}, {niceify(min_agreement_models[0])} vs. {niceify(min_agreement_models[1])}' )
     print (f'max agreement: {max_agreement:.1%}, {niceify(max_agreement_models[0])} vs. {niceify(max_agreement_models[1])}' )
     print (f'mean agreement: {np.nanmean(heatmap):.1%}')

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
                 annot=True, fmt='.2f', cmap='PiYG', vmin=0, vmax=1, center=0.5, square=True, linewidth=1.0,
                 ax = heatmap_ax, cbar_ax=cbar_ax, cbar_kws={'orientation':'horizontal','ticks':[0,0.25,0.5,0.75,1.0]},
                 annot_kws={'fontsize':6})
     heatmap_ax.xaxis.set_ticks_position('top')
     heatmap_ax.tick_params('x', labelrotation=90)
     heatmap_ax.tick_params(axis='both', which='major', labelsize=8)
     heatmap_ax.tick_params(axis='both', which='minor', labelsize=8)
     cbar_ax.tick_params(axis='both', which='major', labelsize=8)
     cbar_ax.tick_params(axis='both', which='minor', labelsize=8)
     cbar_ax.set_xlabel('between model agreement rate\n(proportion of sentence pairs)',fontdict={'fontsize':axes_label_fontsize})
     heatmap_ax.set_ylabel('model 1',fontdict={'fontsize':axes_label_fontsize})
     heatmap_ax.set_xlabel('model 2',fontdict={'fontsize':axes_label_fontsize})
     heatmap_ax.xaxis.set_label_position('top')

     print(f"figure size: {fig_w},{fig_h} inches")

     if save_folder is not None:
               pathlib.Path(save_folder).mkdir(parents=True,exist_ok=True)
               fig.savefig(os.path.join(save_folder,f'{trial_type}_model_by_model_agreement_matrix.pdf'), dpi=600)
     else:
          plt.show()

def data_preprocessing(results_csv = 'behavioral_results/contstim_Aug2021_n100_results.csv'):
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

     return df

def optimization_illustration(df, model1 = 'electra', model2 = 'gpt2', s1='Diddy has a wealth of experience with grappling',s2='Nothing has a world of excitement and joys', n='Luke has a ton of experience with winning', percentile_mode=False, tick_label_fontsize=8, axes_label_fontsize = 8, panel_letter = None,
n_max_chars=None, s1_max_chars=None, s2_max_chars=None):

     matplotlib.rcParams.update({'font.size': 10})
     matplotlib.rcParams.update({'font.family':'sans-serif'})
     matplotlib.rcParams.update({'font.sans-serif':'Arial'})

     m1_log_prob = np.load(f'natural_sentence_probabilities/sents_reddit_natural_June2021_filtered_probs_{model1}.npy')
     m2_log_prob = np.load(f'natural_sentence_probabilities/sents_reddit_natural_June2021_filtered_probs_{model2}.npy')

     # m1_log_prob=scipy.stats.rankdata(m1_log_prob)
     # m1_log_prob=m1_log_prob/m1_log_prob.max()
     # m2_log_prob=scipy.stats.rankdata(m2_log_prob)
     # m2_log_prob=m2_log_prob/m2_log_prob.max()
     def rank_p_vec(p_vec):
          p_vec=pd.Series(p_vec).rank(method='dense')
          p_vec=p_vec/p_vec.max()*100.0
          return np.asarray(p_vec)

     def rank_p_scalar(p_vec, p):
          return np.mean(p>=p_vec)*100.0

     if percentile_mode:
          fig = plt.figure(figsize=(3.25,2.5))
          plt.scatter(x = rank_p_vec(m1_log_prob)[:500], y=rank_p_vec(m2_log_prob)[:500], color=natural_sentence_color, s=5, edgecolors='k', linewidths=0.25)
     else:
          fig = plt.figure(figsize=(5,5))
          plt.scatter(x = m1_log_prob[:500], y=m2_log_prob[:500], color=natural_sentence_color,s=5, linewidths=0.25)
     ax = plt.gca()

     ax.set_aspect('equal')

     if percentile_mode:
          plt.xlabel(niceify(model1)+'\n'+'p(sentence) percentile', fontdict={'fontsize':axes_label_fontsize})
          plt.ylabel(niceify(model2)+'\n'+'p(sentence) percentile', fontdict={'fontsize':axes_label_fontsize})
     else:
          plt.xlabel(niceify(model1)+'\n'+'$log\,p(sentence)$', fontdict={'fontsize':axes_label_fontsize})
          plt.ylabel(niceify(model2)+'\n'+'$log\,p(sentence)$', fontdict={'fontsize':axes_label_fontsize})

     def get_sentence_prob(df,sentence,model):
          df2=df[df['sentence1']==sentence]
          if len(df2)>0:
               return df2[f'sentence1_{model}_prob'].mean()
          df2=df[df['sentence2']==sentence]
          if len(df2)>0:
               return df2[f'sentence2_{model}_prob'].mean()
          return None

     log_p_s1_model1 = get_sentence_prob(df,s1,model1)
     log_p_s1_model2 = get_sentence_prob(df,s1,model2)
     log_p_s2_model1 = get_sentence_prob(df,s2,model1)
     log_p_s2_model2 = get_sentence_prob(df,s2,model2)
     log_p_n_model1 = get_sentence_prob(df,n,model1)
     log_p_n_model2 = get_sentence_prob(df,n,model2)

     caption_text =  inspect.cleandoc(
          f'''In this example, we start with the randomly sampled natural sentence
          ``{n}''. If we adjust this sentence to minimize its probability according to
          {niceify(model1)} (while keeping the sentence at least as likely as the natural
          sentence according to {niceify(model2)}), we obtain the synthetic sentence
          ``{s1}''. This adjustment decreases {niceify(model1)}'s log probability
          (by {log_p_n_model1-log_p_s1_model1:.1f}),
          but has little effect on {niceify(model2)}'s log probability (increasing it
          by {log_p_s1_model2-log_p_n_model2:.1f}). By repeating this procedure while
          switching the roles of the models, we generate
          the synthetic sentence ``{s2}'', which decreases {niceify(model2)}'s log
          probability (by {log_p_n_model2-log_p_s2_model2:.1f}), while slightly
          increasing {niceify(model1)}'s (by {log_p_s2_model1-log_p_n_model1:.1f}).''').replace('\n',' ')

     if panel_letter is not None:
          caption_text = f'\\boldbf({panel_letter}) ' + caption_text
     print(caption_text)

     if percentile_mode:
          log_p_s1_model1 = rank_p_scalar(m1_log_prob,log_p_s1_model1)
          log_p_s1_model2 = rank_p_scalar(m2_log_prob,log_p_s1_model2)
          log_p_s2_model1 = rank_p_scalar(m1_log_prob,log_p_s2_model1)
          log_p_s2_model2 = rank_p_scalar(m2_log_prob,log_p_s2_model2)
          log_p_n_model1 = rank_p_scalar(m1_log_prob,log_p_n_model1)
          log_p_n_model2 = rank_p_scalar(m2_log_prob,log_p_n_model2)

     ax.plot(log_p_s1_model1,log_p_s1_model2,markersize=6,markerfacecolor=synthetic_sentence_color,markeredgecolor='k',zorder=1000,marker='o')
     ax.plot(log_p_s2_model1,log_p_s2_model2,markersize=6,markerfacecolor=synthetic_sentence_color,markeredgecolor='k',zorder=1000,marker='o')
     ax.plot(log_p_n_model1,log_p_n_model2,markersize=6,markerfacecolor=natural_sentence_color,markeredgecolor='k',zorder=1000,marker='o')

     arrow_style = mpatches.ArrowStyle.Simple() #(head_length=1.0, head_width=1.0, tail_width=.4)
     arrow = mpatches.FancyArrowPatch((log_p_n_model1, log_p_n_model2), (log_p_s2_model1, log_p_s2_model2),
                                 mutation_scale=10,facecolor='k',edgecolor='k',arrowstyle=arrow_style,  shrinkA=6, shrinkB=6, linewidth=0.5)
     ax.add_patch(arrow)
     arrow = mpatches.FancyArrowPatch((log_p_n_model1, log_p_n_model2), (log_p_s1_model1, log_p_s1_model2),
                                 mutation_scale=10,facecolor='k',edgecolor='k',arrowstyle=arrow_style,  shrinkA=6, shrinkB=6, linewidth=0.5)
     ax.add_patch(arrow)

     def break_lines(s, max_chars):
          if max_chars is None:
               return s
          else:
               return textwrap.fill(s, width=max_chars, break_long_words=False, drop_whitespace=True, break_on_hyphens=False)
     path_effects = [] #[PathEffects.withStroke(linewidth=1, foreground='w')]
     ax.annotate(text=break_lines(s1,s1_max_chars)+'.',xy=(log_p_s1_model1,log_p_s1_model2),horizontalalignment='left',va='bottom', xytext=(-3,4),textcoords='offset points', path_effects=path_effects,fontsize=7,fontfamily='sans-serif', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round', pad=0.05))
     ax.annotate(text=break_lines(s2,s2_max_chars)+'.',xy=(log_p_s2_model1,log_p_s2_model2),horizontalalignment='left',va='top', xytext=(6,3),textcoords='offset points',path_effects=path_effects,fontsize=7,fontfamily='sans-serif')
     ax.annotate(text=break_lines(n,n_max_chars)+'.',xy=(log_p_n_model1,log_p_n_model2),horizontalalignment='left',va='top', xytext=(6,3),textcoords='offset points',path_effects=path_effects,fontsize=7,fontfamily='sans-serif', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round', pad=0.05))

     # print(model1,log_p_s1_model1,model2,log_p_s1_model2,s1)
     # print(model1,log_p_s2_model1,model2,log_p_s2_model2,s2)
     # print(model1,log_p_n_model1,model2,log_p_n_model2,n)

     ax.spines['right'].set_visible(False)
     ax.spines['top'].set_visible(False)

     delta_m1 = log_p_s2_model1-log_p_s1_model1
     delta_m2 = log_p_s1_model2-log_p_s2_model2
     if percentile_mode:
          plt.xlim([0,100])
          plt.ylim([0,100])
     else:
          delta_m1 = log_p_s2_model1-log_p_s1_model1
          delta_m2 = log_p_s1_model2-log_p_s2_model2
          space_factor = 1/2
          plt.xlim([log_p_s1_model1-delta_m1*space_factor,log_p_s2_model1+delta_m1*space_factor])
          plt.ylim([log_p_s2_model2-delta_m1*space_factor,log_p_s1_model2+delta_m2*space_factor])

     ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
     ax.tick_params(axis='both', which='minor', labelsize=tick_label_fontsize)

     if percentile_mode:
          suffix = '_percentile'
     else:
          suffix = '_logprob'
     plt.subplots_adjust(left=0.175, right=0.69, top=1.0, bottom=0.1)

     if panel_letter is not None:
          plt.figtext(x=0, y=1,
                         s=panel_letter, verticalalignment='top',
                         fontdict={'fontsize':panel_letter_fontsize,'weight':'bold'}
                         )
     pathlib.Path('figures/optimization_illustration').mkdir(exist_ok=True, parents=True)
     plt.savefig(os.path.join('figures','optimization_illustration',f'optimization_illustration_{n}{suffix}.pdf'))

if __name__ == '__main__':

     df = data_preprocessing()

     # Figure 2
     # optimization_illustration(df,model1='gpt2',model2='electra',s2='Diddy has a wealth of experience with grappling',s2_max_chars=32, s1='Nothing has a world of excitement and joys', s1_max_chars = 11, n='Luke has a ton of experience with winning', n_max_chars = 28, percentile_mode=True, panel_letter='a')
     # optimization_illustration(df,model1='roberta',model2='trigram',s1='You have to realize is that noise again',s1_max_chars=19, s2='I wait to see how it shakes out', s2_max_chars=17, n='I need to see how this played out', n_max_chars=17, percentile_mode=True, panel_letter='b')

     # # %% Binarized accuracy measurements

     # # # uncomment to plot main result figures
     # figs=plot_main_results_figures(df, save_folder = 'figures/binarized_acc',measure='binarized_accuracy')
     # #plt.show()

     # # warning - this is a slow one to run (an hour or so)
     figs=plot_main_results_figures(df, measure='RAE_signed_rank_cosine_similarity',save_folder = 'figures/RAE_signed_rank_cosine_similarity_analytical_new_LB')
     plt.show()

     # Figure S3
     # model_by_model_agreement_heatmap(df,save_folder='figures/heatmaps', trial_type = 'randomly_sampled_natural')
     # plt.show()

     # uncomment to plot model by model accuracy heatmaps (Figure S4)
     # for trial_type in ['natural_controversial','synthetic_vs_synthetic']:
     #      model_by_model_consistency_heatmap(df,trial_type = trial_type, save_folder = 'figures/heatmaps')

     # uncomment to plot n_vs_s heatmap (Figure S5)
     # model_by_model_N_vs_S_heatmap(df,save_folder='figures/heatmaps')

     # individual sentence probability scatter plots (left-hand panels in Figures 1 and 3)
     # x_model='gpt2'
     # y_model='roberta'
     # # sentence_pair_scatter_plot(df, x_model=x_model, y_model=y_model, trial_type='randomly_sampled_natural', targeting_1=None, targeting_2=None, targeted_model_1=None,targeted_model_2=None)
     # # sentence_pair_scatter_plot(df, x_model=x_model, y_model=y_model, trial_type='natural_controversial', targeting_1=None, targeting_2=None, targeted_model_1=x_model,targeted_model_2=y_model)
     # sentence_pair_scatter_plot(df, x_model=x_model, y_model=y_model, trial_type='natural_vs_synthetic', targeting_1='accept', targeting_2='reject', targeted_model_1=x_model,targeted_model_2=y_model)
     # plt.savefig(f'figures/sentence_scatterplots/natural_vs_synthetic_{x_model}_accept_vs_{y_model}_reject.pdf')
     # sentence_pair_scatter_plot(df, x_model=x_model, y_model=y_model, trial_type='natural_vs_synthetic', targeting_1='reject', targeting_2='accept', targeted_model_1=x_model,targeted_model_2=y_model)
     # plt.savefig(f'figures/sentence_scatterplots/natural_vs_synthetic_{x_model}_reject_vs_{y_model}_accept.pdf')
     # sentence_pair_scatter_plot(df, x_model=x_model, y_model=y_model, trial_type='synthetic_vs_synthetic', targeting_1=None, targeting_2=None, targeted_model_1=x_model,targeted_model_2=y_model)
     # plt.savefig(f'figures/sentence_scatterplots/synthetic_vs_synthetic_{x_model}_vs_{y_model}.pdf')

     # Tables 1-3:
     # generate_worst_sentence_pairs_table(df, trial_type  = 'natural_controversial', n_sentences_per_model=1)
     # generate_worst_sentence_pairs_table(df, trial_type  = 'synthetic_vs_synthetic', n_sentences_per_model=1)
     # generate_worst_sentence_pairs_table(df, trial_type  = 'natural_vs_synthetic', n_sentences_per_model=1, targeting='accept')

     # uncomment this next line to generate detailed html result tables
     # build_all_html_files(df)

     # deprecated analyses
     # figs=plot_main_results_figures(df, save_folder = 'figures/SomersD',measure='SomersD')
     # figs=plot_main_results_figures(df, save_folder = 'figures/Flexible_SomersD',measure='flexible_SomersD')
     # plt.show()
