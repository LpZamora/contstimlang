# %% Setup and load data
import glob, os
from operator import itruediv
import itertools

import numpy as np
import scipy.stats
import gurobipy  as gb # Gurobi requires installing a license first (there's a free academic license)
from gurobipy import GRB
import pandas as pd

npys = glob.glob(os.path.join('natural_sentence_probabilities','*.npy'))
models = [s.split('.')[-2].split('_')[-1] for s in npys]
n_models=len(models)

txt_fname='sents_reddit_natural_May2021_filtered.txt'
with open(txt_fname) as f:
    sentences = f.readlines()
sentences = [s.strip() for s in sentences]
n_sentences=len(sentences)

log_prob=np.zeros((n_sentences,n_models))
log_prob[:]=np.nan

for i_model, (model, npy) in enumerate(zip(models,npys)):
    log_prob[:,i_model] = np.load(npy)

# %% rank sentence probabilities within each model
ranked_log_prob = scipy.stats.rankdata(log_prob,axis=0, method='average')/len(log_prob)

# %% throw sentences that are not controversial at all
def prefilter_sentences(sentences, ranked_log_prob):
    # prefilter sentences for controversiality according to at least one model pair
    mask=np.zeros((n_sentences,),dtype=bool)
    for i_model in range(n_models):
        for j_model in range(n_models):
            if i_model==j_model:
                continue

            p1 = ranked_log_prob[:,i_model]
            p2 = ranked_log_prob[:,j_model]

            cur_mask = np.logical_and(p1 <  0.5, p2 >= 0.5)

            mask = np.logical_or(mask,cur_mask)

    return list(np.asarray(sentences)[mask]), ranked_log_prob[mask]
sentences, ranked_log_prob = prefilter_sentences(sentences, ranked_log_prob)
n_sentences = len(sentences)
print('prefiltered the sentences to a total of',n_sentences,'sentences.')

# # %%  Use Gurobipy to select controversial sentences
def select_sentences_Gurobi(sentences, models, ranked_log_prob, n_trials_per_pair=10, mode='minsum'):
    """ Select controversial sentence pairs using mixed integer linear programming.
        This function selects pairs of sentences s1 and s2 for multiple model pairs m1 and m2
        to minimize sum_i[r(s1_i|m1_i)+r(s2_i|m2_i)] (i indexes trials)
        s.t.,
        r(s1_i|m2_i)>=0.5,
        r(s2_i|m1_i)>=0.5,
        and no sentence is used more than once.

        mode='minmax' aggregates trials by the maximum function instead of summation, but it results
        in less controversial sentences.
    """
    n_sentences = len(sentences)
    assert len(ranked_log_prob)==n_sentences

    n_models = ranked_log_prob.shape[1]
    assert len(models) == n_models

    assert mode == 'minsum' or mode == 'minmax'

    model_1 = []
    model_2 = []
    for i_model, j_model in itertools.combinations(range(n_models),2):
        for i_trial in range(n_trials_per_pair):
            model_1.append(i_model)
            model_2.append(j_model)

    n_trials = len(model_1)

    m = gb.Model()

    X = m.addMVar((n_sentences,n_trials,2),vtype=GRB.BINARY)

    if mode == 'minsum':
        loss = 0
    elif mode == 'minmax':
        loss = m.addMVar((1,))

    for i_trial in range(n_trials):

        # Evaluate the rank probability of the selected sentences by calculating the
        # dot-product between the vector of sentence rank probabilities and a binary
        # vector, all 0 except 1.0 for the selected sentence for i_trial.

        s1_m1=ranked_log_prob[:,model_1[i_trial]].T @ X[:,i_trial,0] # the ranked probability of s1 according to model 1 (scalar)
        s1_m2=ranked_log_prob[:,model_2[i_trial]].T @ X[:,i_trial,0]
        s2_m1=ranked_log_prob[:,model_1[i_trial]].T @ X[:,i_trial,1]
        s2_m2=ranked_log_prob[:,model_2[i_trial]].T @ X[:,i_trial,1]

        # we want s1_m1 and s2_m2 to be small, and s1_m2 and s2_m1 to be big.
        if mode =='minsum':
            loss+= (s1_m1 + s2_m2)
        elif mode=='minmax':
            m.addConstr(loss >= s1_m1)
            m.addConstr(loss >= s2_m2)

        m.addConstr(s1_m2 >= 0.5)
        m.addConstr(s2_m1 >= 0.5)

        # each trial should have one s1 and one s2.
        m.addConstr(X[:,i_trial,0].sum()==1)
        m.addConstr(X[:,i_trial,1].sum()==1)

    # only sentence should be used not more than once.
    for i_sentence in range(n_sentences):
        m.addConstr(X[i_sentence].sum()<=1)

    m.setObjective(loss, GRB.MINIMIZE)
    m.update()

    m.optimize()

    print('Obj: %g' % m.objVal)
    assert m.status == gb.GRB.Status.OPTIMAL

    solution = X.X
    # extracting solution
    df = []
    for i_trial in range(n_trials):
        s1_idx=np.flatnonzero(solution[:,i_trial,0])[0]
        s2_idx=np.flatnonzero(solution[:,i_trial,1])[0]
        d={'sentence1':sentences[s1_idx],
           'sentence2':sentences[s2_idx],
           'model_1':models[model_1[i_trial]],
           'model_2':models[model_2[i_trial]],
           }
        d['s1_ranked_log_prob_model_1']=ranked_log_prob[s1_idx,model_1[i_trial]]
        d['s1_ranked_log_prob_model_2']=ranked_log_prob[s1_idx,model_2[i_trial]]
        d['s2_ranked_log_prob_model_1']=ranked_log_prob[s2_idx,model_1[i_trial]]
        d['s2_ranked_log_prob_model_2']=ranked_log_prob[s2_idx,model_2[i_trial]]
        df.append(d)
    df = pd.DataFrame(df)
    return df

#X=minmax_opt(sentences,models,ranked_log_prob)
df=select_sentences_Gurobi(sentences,models,ranked_log_prob,mode='minsum')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)
print(df)

df.to_csv(txt_fname.replace('_filtered.txt','_selected.csv'))