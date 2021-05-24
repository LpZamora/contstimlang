# %%
from collections import OrderedDict
import random
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

import pandas as pd
import scipy.stats

#from isotonic_response_model import overfitted_isotonic_mapping

results_csv = 'behavioral_results/contstim_N32_results.csv'

# %% preprocess results
def align_sentences(df):
     """ To ease analysis, we align all trials so the order of sentences
     within each sentence pair is lexicographical rather than based on display position.
     This ensures that different subjects can be directly compared to each other.

     This script also changes creates a numerical "rating" column, with 1 = strong preference for sentence1, 6 = strong preference for sentence2.

     TODO: sentence1_type and sentence2_type might be jumbled, I'm not yet sure.
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

aligned_results_csv = results_csv.replace('.csv','_aligned.csv')
try:
     df=pd.read_csv(aligned_results_csv)
except:
     df=pd.read_csv(results_csv)
     df=align_sentences(df)
     df.to_csv(aligned_results_csv)

# remove excluded subjects
excluded_subjects=[3572610,3572431]
df=df[~df['Participant Private ID'].isin(excluded_subjects)]


# %% leave one subject out noise ceiling
def add_leave_one_subject_predictions(df):
     """ Leave one subject out noise ceiling
     All of the following measures are lower bounds on the noise ceiling.
     In other words, an ideal model should be at least as good as these measures.
     """

     # add a sentence pair field. This assumes that s1 and s2 are already sorted.
     df['sentence_pair']=[s1 + '_' + s2 for s1, s2 in zip(df['sentence1'],df['sentence2'])]

     # The LOOSO loop.
     df2=df.copy()

     df2['binarized_choice_probability_NC']=np.nan
     df2['majority_vote_NC']=np.nan
     df2['mean_rating_NC']=np.nan

     # def assign(df, index, field,val):
     #      df.iloc[index,df.columns.get_loc(field)]=val
     #      print(df.iloc[index,df.columns.get_loc(field)])

     for trial_idx, trial in tqdm(df.iterrows(),total=len(df),desc='leave one subject out NC calculation.'):
          # choose all trials with the same sentence pair in OTHER subjects.
          mask=(df['sentence_pair']==trial['sentence_pair']) \
               & (df['Participant Private ID']!=trial['Participant Private ID'])
          reduced_df=df[mask]

          # we add three kinds of noise ceilings:

          # 1. binarized choice probability:
          # the predicted probability that a subject will prefer sentence2
          # (to be used for binomial prediction)
          df2.loc[trial_idx,'binarized_choice_probability_NC']=(reduced_df['rating']>=4).mean()

          # 2. simple majority vote (1: sentence2, 0: sentence1)
          # to be used for accuracy evaluation)
          if df2.loc[trial_idx,'binarized_choice_probability_NC']>0.5:
               df2.loc[trial_idx,'majority_vote_NC']=1
          elif df2.loc[trial_idx,'binarized_choice_probability_NC']<0.5:
               df2.loc[trial_idx,'majority_vote_NC']=0
          else:
               raise Warning(f'Tied predictions for trial {trial_idx}. Randomzing prediction.')
               df2.loc[trial_idx,'majority_vote_NC']=random.choice([0,1])

          # 3. And last, we simply average the ratings
          # to be used for correlation based measures
          df2.loc[trial_idx,'mean_rating_NC']=(reduced_df['rating']).mean()
     return df2

aligned_results_csv_with_loso = results_csv.replace('.csv','_aligned_with_loso.csv')
try:
     df=pd.read_csv(aligned_results_csv_with_loso)
except:
     df=add_leave_one_subject_predictions(df)
     df.to_csv(aligned_results_csv_with_loso)

# %% Binarized accuracy measurements
def get_binarized_accuracy(df,models):
     df2=df.copy()

     """ binarizes model and human predictions and returns 1 or 0 for prediction correctness """
     for model in models:
          if model != 'majority_vote_NC':
               assert not (df['sentence2_'+model + '_prob']==df['sentence1_'+model + '_prob']).any(), f'found tied prediction for model {model}'
               model_predicts_sent2=df['sentence2_'+model + '_prob']>df['sentence1_'+model + '_prob']
               human_chose_sent2=df['rating']>=4
               df2[model]=model_predicts_sent2==human_chose_sent2
               df2=df2.drop(columns=['sentence1_'+model + '_prob','sentence2_'+model + '_prob'])
          else: # deal with human NC
               df2['majority_vote_NC']=df2['majority_vote_NC']==human_chose_sent2
     return df2

def get_models(df):
     """ a helper function for extracting model names from column names """
     models = [re.findall('sentence1_(.+)_prob',col)[0] for col in df.columns if re.search('sentence1_(.+)_prob',col)]
     return models
models = get_models(df)+['majority_vote_NC']
df_acc=get_binarized_accuracy(df,models)

def average_models_within_conditions(df, models, targeting=None, pair_type=None):
     df = df.copy()
     df['pair_type']=['N_vs_S' if t1=='N' or t2=='N' else 'S_vs_S' for t1,t2 in zip(df.sentence1_type, df.sentence2_type)]

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

          cur_result['mean']=reduced_df[model].mean()
          cur_result['se']=reduced_df[model].std()/np.sqrt(len(reduced_df[model]))

          NC_measures = [col for col in df.columns if col.endswith('_NC')]
          for NC_measure in NC_measures:
               cur_result[NC_measure]=reduced_df[NC_measure].mean()
          results.append(cur_result)
     return pd.DataFrame(results)

print ('all trials:')
print(df_acc[models].mean(axis = 0, skipna = True))

print ('only targeted:')
models = get_models(df)
df_acc_targeted = average_models_within_conditions(df_acc, models, targeting='targeted')
print(df_acc_targeted)

print ('only untargeted:')
models = get_models(df)
df_acc_targeted = average_models_within_conditions(df_acc, models, targeting='untargeted')
print(df_acc_targeted)




# %% Correlation based measurements
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


# %% Correlation based measurements
def get_correlation_based_accuracy(df,models, correlation_func='pearsonr'):
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


score_df = log_prob_pairs_to_scores(df, transformation_func='diff')
models = get_models(df)+['mean_rating_NC']
corr_df=get_correlation_based_accuracy(score_df,models, correlation_func='pearsonr')[models]
print('predicting ratings from log-prob differences (pearson)')
print(corr_df.mean(axis = 0, skipna = True))

corr_df=get_correlation_based_accuracy(score_df,models,  correlation_func='spearmanr')[models]
print('predicting ratings from log-prob differences (spearman)')
print(corr_df.mean(axis = 0, skipna = True))

corr_df=get_correlation_based_accuracy(score_df,models, correlation_func='kendall-Tau-b')[models]
print('predicting ratings from log-prob differences (kendall-Tau-b)')
print(corr_df.mean(axis = 0, skipna = True))
