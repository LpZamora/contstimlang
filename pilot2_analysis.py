# %%
from collections import OrderedDict
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import pandas as pd

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
     """ leave one subject out noise ceiling """

     # add a sentence pair field. This assumes that s1 and s2 are already sorted.
     df['sentence_pair']=[s1 + '_' + s2 for s1, s2 in zip(df['sentence1'],df['sentence2'])]

     # The LOOSO loop.
     df2=df.copy()

     df2['binarized_choice_probability']=np.nan
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
          df2.loc[trial_idx,'binarized_choice_probability']=(reduced_df['rating']>=4).mean()

          # 2. simple majority vote (1: sentence2, 0: sentence1)
          # to be used for accuracy evaluation)
          if df2.loc[trial_idx,'binarized_choice_probability']>0.5:
               df2.loc[trial_idx,'majority_vote_NC']=1
          elif df2.loc[trial_idx,'binarized_choice_probability']<0.5:
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
models = ['gpt2','roberta','electra','bert','xlm','lstm','rnn','trigram','bigram','majority_vote_NC']
def get_binarized_correctness(df,models):
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

df2=get_binarized_correctness(df,models)[models]
print(df2.mean(axis = 0, skipna = True))