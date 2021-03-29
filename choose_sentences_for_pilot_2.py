import os, glob
import itertools
import re

import numpy as np
import pandas as pd
import scipy.stats

def move_cols_to_left(df,cols_to_move):
    """ reorder dataframe's columns such that the block of columns specified by the list cols_to_move is at the left """
    cols = list(df.columns)
    cols = cols_to_move + [col for col in cols if col not in cols_to_move]
    return df[cols]

synthesized_sentence_csv_folder = 'synthesized_sentences/20210224_controverisal_sentence_pairs_heuristic_natural_init_allow_rep/8_word'
output_csv_path = 'synthesized_sentences/20210224_controverisal_sentence_pairs_heuristic_natural_init_allow_rep/8_word'
n_sentence_triplets_for_consideration=100
n_sentence_triplets_for_human_testing=10
selection_method = 'decile_balanced' # 'decile_balanced' or 'best'

def docplex_solve(utility_vec,grouping1,grouping2,n_selected_per_group,maximize=True):
    """ selects elements from utility_vec such that each group in grouping1 and each group in grouping 2 are sampled exactly n_selected_per_group times
    utility vec (np.ndarray or list) (n,) long vector of utility values
    grouping 1 (np.ndarray or list) (n,) long vector or list of groups
    grouping 2 (np.ndarray or list) (n,) long vector or list of groups
    n_selected_per_group (int) how many samples per marginal group
    maximize (bool) maximize utility if True, minimize if False

    returns list of selected elements
    """
    from docplex.mp.model import Model # requires also cplex -  conda install -c ibmdecisionoptimization cplex

    mdl = Model()

    n = len(utility_vec)
    assert (len(grouping1)==n) and (len(grouping2)==n)

    groups1 = np.unique(grouping1)
    groups2 = np.unique(grouping2)

    assert (len(groups1)*n_selected_per_group)<=n and (len(groups2)*n_selected_per_group)<=n

    selection = mdl.binary_var_list(n, name='s') # this represents whether each element was "selected" or not
    utility_vec=list(utility_vec)

    utility=mdl.sum(utility_vec[i] * selection[i] for i in range(n))

    mdl.add_constraints([
        mdl.sum(selection[i] for i in range(n) if grouping1[i]==g) == n_selected_per_group
        for g in groups1])

    mdl.add_constraints([
        mdl.sum(selection[i] for i in range(n) if grouping2[i]==g) == n_selected_per_group
        for g in groups2])

    if maximize:
        mdl.maximize(utility)
    else:
        mdl.minimize(utility)

    assert mdl.solve()

    solution=np.zeros(n,dtype=bool)
    for i in range(n):
        solution[i]=selection[i].solution_value==1
    return solution


def _test_docplex_solve():
    """ a simple sanity test for docplex_solve() """
    n=100
    utility_vec = np.random.rand(n,)
    grouping1 = np.random.permutation(100) % 10
    grouping2 = np.random.permutation(100) % 10
    print('maximize:')
    solution = docplex_solve(utility_vec,grouping1,grouping2,1,maximize=True)
    print('grouping 1:',grouping1[solution])
    print('grouping 2:',grouping2[solution])
    print(np.sum(utility_vec[solution]))
    assert len(np.unique(grouping1[solution]))==10
    assert len(np.unique(grouping2[solution]))==10

    print('minimize:')
    solution = docplex_solve(utility_vec,grouping1,grouping2,1,maximize=False)
    print('grouping 1:',grouping1[solution])
    print('grouping 2:',grouping2[solution])
    print(np.sum(utility_vec[solution]))
    assert len(np.unique(grouping1[solution]))==10
    assert len(np.unique(grouping2[solution]))==10
#_test_docplex_solve()

# start by checking which model pairs are available, and loading csvs into pandas dataframes
csvs = glob.glob(os.path.join(synthesized_sentence_csv_folder,'*.csv'))
all_model_names = set()
df_dict={}
for csv in csvs:
    model1,model2 = re.findall(r'(.+)_vs_(.+)\.csv',os.path.basename(csv))[0]
    all_model_names.add(model1)
    all_model_names.add(model2)

    cur_df = pd.read_csv(csv,names = ['N_idx','N','S','loss','lp_N_given_m1','lp_S_given_m1','lp_N_given_m2','lp_S_given_m2']).sort_index()

    # due to some glitch in the cluster task scheduling (and one duplicate sentence in the natural sentece list),
    # a small number of natural sentences were used more than once per ordered model pair
    # keep only the first one (we don't keep the best to avoid biasing the sentence sample)
    cur_df = cur_df.drop_duplicates(subset='N')

    cur_df['m1'] = [model1]*len(cur_df)
    cur_df['m2'] = [model2]*len(cur_df)
    df_dict[(model1,model2)] = cur_df

# processs each model pair ###########

all_sentences=[]
selected_sentences=[]

# next, we filter out sentence whose natural sentence (N) doesn't appear in to CSVs related to each model pair comparison and join dataframes
for model1, model2 in itertools.combinations(all_model_names,2):

    df1 = df_dict[(model1,model2)]
    df2 = df_dict[(model2,model1)]

    # to prepare dataframes for merging, we rename the columns. The two dataframes will be consistent in terms of which models are labeled as m1 and m2
    df1 = df1.rename(columns={'S':'S1','loss':'loss_S1_vs_N','lp_S_given_m1':'lp_S1_given_m1','lp_S_given_m2':'lp_S1_given_m2'})
    df2 = df2.rename(columns={'S':'S2','loss':'loss_S2_vs_N','lp_S_given_m1':'lp_S2_given_m2','lp_S_given_m2':'lp_S2_given_m1','lp_N_given_m1':'lp_N_given_m2','lp_N_given_m2':'lp_N_given_m1','m1':'m2','m2':'m1'})

    # merging dataframes
    cur_df = df1.merge(df2,how='inner',on=['N_idx','N','m1','m2'],sort=True,validate='one_to_one',suffixes=['','_prime'])

    cur_df = move_cols_to_left(cur_df,['m1','m2','N_idx','N','S1','S2']) # reorder columns for readability

    # a sanity check:
    r = scipy.stats.pearsonr(cur_df['lp_N_given_m1'].to_numpy(),cur_df['lp_N_given_m1_prime'].to_numpy())[0]
    assert r>.999,'mismatching natural sentence probability for ' +model1 + ' in ' + model1 +'_vs_' + model2
    r = scipy.stats.pearsonr(cur_df['lp_N_given_m2'].to_numpy(),cur_df['lp_N_given_m2_prime'].to_numpy())[0]
    assert r>.999,'mismatching natural sentence probability for '+ model2 + ' in ' + model1 +'_vs_' + model2
    cur_df = cur_df.drop(columns=['lp_N_given_m1_prime','lp_N_given_m2_prime'])

    # We consider 99 (TODO - 100) natural sentences per model pair
    assert len(cur_df)>=n_sentence_triplets_for_consideration
    cur_df = cur_df.head(n=n_sentence_triplets_for_consideration)

    # now, we select trials for the human experiment.
    # rank loss_S1_vs_N and loss_S2_vs_N
    cur_df['loss_rank1'] = cur_df['loss_S1_vs_N'].rank(ascending=True)
    cur_df['loss_rank2'] = cur_df['loss_S2_vs_N'].rank(ascending=True)

    cur_df['worst_rank_of_two'] = np.maximum(cur_df['loss_rank1'],cur_df['loss_rank2'])
    cur_df['rank'] = cur_df['worst_rank_of_two'].rank(ascending=True,method='first')

    if selection_method =='best':
        # just use the most controversial triplets, according to their worst rank across two synthetic sentences

        cur_df['selected_for_human_testing'] = cur_df['rank'] <= n_sentence_triplets_for_human_testing

    elif selection_method == 'decile_balanced':
        # choose the most controversial sentences, under the constraint of equally sampling the natural sentence probability deciles of each model

        # rank loss_S1_vs_N and loss_S2_vs_N
        cur_df['p_N_given_m1_decile']=pd.qcut(cur_df['lp_N_given_m1'],q=10,labels=False)
        cur_df['p_N_given_m2_decile']=pd.qcut(cur_df['lp_N_given_m2'],q=10,labels=False)

        cur_df['selected_for_human_testing'] = docplex_solve(utility_vec=cur_df['rank'],
        grouping1=cur_df['p_N_given_m1_decile'],
        grouping2=cur_df['p_N_given_m2_decile'],
        n_selected_per_group=1,maximize=False)

    cur_df = cur_df.drop(columns='rank')
    selected_df=cur_df[cur_df['selected_for_human_testing']]

    # some more sanity checks
    assert len(selected_df)==n_sentence_triplets_for_human_testing

    # model 2 scores S1 as less natural than N, while model 1 doesn't.
    assert (selected_df['lp_S1_given_m2'] < selected_df['lp_N_given_m2']).all()
    assert (selected_df['lp_S1_given_m1'] >= selected_df['lp_N_given_m1']).all()

    # model 1 scores S2 as less natural than N, while model 2 doesn't.
    assert (selected_df['lp_S2_given_m1'] < selected_df['lp_N_given_m1']).all()
    assert (selected_df['lp_S2_given_m2'] >= selected_df['lp_N_given_m2']).all()

    all_sentences.append(cur_df)
    selected_sentences.append(selected_df)

all_sentences = pd.concat(all_sentences,axis=0)
selected_sentences = pd.concat(selected_sentences,axis=0)

all_sentences.to_csv(output_csv_path +'_' + str(len(all_model_names)) + '_models_100_sentences_per_pair.csv', index=False)
selected_sentences.to_csv(output_csv_path +'_' + str(len(all_model_names)) + '_models_100_sentences_per_pair_best10.csv', index=False)

def all_sentence_lp_plot(all_sentences, figname):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches

    all_model_names = set(all_sentences['m1']) | set(all_sentences['m2'])
    n_models=len(all_model_names)

    inch_per_subplot = 2
    inch_per_margin=0.25
    fig=plt.figure(figsize=(n_models*inch_per_subplot+inch_per_margin*2,n_models*inch_per_subplot+inch_per_margin*2))
    relative_margin = inch_per_margin*2 / (n_models*inch_per_subplot+inch_per_margin*2)
    gs=gridspec.GridSpec(nrows=n_models,ncols=n_models,figure=fig,left=relative_margin,right=1-relative_margin,bottom=relative_margin,top=1-relative_margin,wspace=0.4,hspace=0.4)

    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6

    for c, x_model in enumerate(all_model_names):
        for r, y_model in enumerate(all_model_names):
            if x_model == y_model:
                continue

            fig.add_subplot(gs[r,c])
            mask = (((all_sentences['m1']==x_model) & (all_sentences['m2']==y_model)) |
                      ((all_sentences['m1']==y_model) & (all_sentences['m2']==x_model)))
            cur_df=all_sentences[mask]

            for i_row, row in cur_df.iterrows():
                if row['m1'] == x_model and row['m2'] == y_model:
                    x_N=row['lp_N_given_m1']
                    y_N=row['lp_N_given_m2']
                    x_S1=row['lp_S1_given_m1']
                    y_S1=row['lp_S1_given_m2']
                    x_S2=row['lp_S2_given_m1']
                    y_S2=row['lp_S2_given_m2']
                elif row['m1'] == y_model and row['m2'] == x_model:
                    x_N=row['lp_N_given_m2']
                    y_N=row['lp_N_given_m1']
                    x_S1=row['lp_S1_given_m2']
                    y_S1=row['lp_S1_given_m1']
                    x_S2=row['lp_S2_given_m2']
                    y_S2=row['lp_S2_given_m1']

                linewidth=0.1

                if row['selected_for_human_testing']:
                    edgecolor='g'
                    zorder=0
                else:
                    edgecolor='r'
                    zorder=-1

                ax=plt.gca()

                plt.plot((x_N, x_S1), (y_N, y_S1),linewidth=linewidth,color=edgecolor,zorder=zorder,alpha=0.5)
                plt.plot((x_N, x_S2), (y_N, y_S2),linewidth=linewidth,color=edgecolor,zorder=zorder,alpha=0.5)
                plt.plot((x_N,), (y_N,),marker='o',linewidth=linewidth,color=edgecolor,fillstyle='full',markersize=1.5,zorder=zorder,alpha=0.5)


            plt.xlabel(x_model,fontsize=8)
            plt.ylabel(y_model,fontsize=8)

            ax.set_xlim(ax.get_xlim()[::-1])
            #ax.set_ylim(ax.get_ylim()[::-1])
            ax.xaxis.set_ticks_position('top') # the rest is the same
            ax.xaxis.set_label_position('top')

    fig.savefig(figname)
    plt.close()
all_sentence_lp_plot(all_sentences, output_csv_path +'_' + str(len(all_model_names)) + '_models_100_sentences_per_pair.pdf' )
