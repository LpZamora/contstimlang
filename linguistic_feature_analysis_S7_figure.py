import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import statsmodels.api as sm
from wordfreq import word_frequency
import cmudict
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.stats as stats
import statsmodels
plt.rcParams['figure.dpi']= 200

def confidence_ellipse(x, y, ax, n_std, facecolor='none', **kwargs):

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)


    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


df=pd.read_csv('behavioral_results/contstim_Aug2021_n100_results_anon_aligned.csv')

model1s=list(df['sentence1_model'])

models=['gpt2','roberta','electra','bert','xlm','lstm','rnn','trigram','bigram']

res_dict={}

for m in models:
    res_dict[m]={}

for i in range(len(model1s)):
    
    if df['sentence1_type'][i]=='S' and df['sentence2_type'][i]=='S':
        
        sent_pair=df['sentence_pair'][i]
        
        m1=df['sentence1_model'][i]
        m2=df['sentence2_model'][i]
        
        pm1s1=df['sentence1_'+m1+'_prob'][i]
        pm2s1=df['sentence1_'+m2+'_prob'][i]
        pm1s2=df['sentence2_'+m1+'_prob'][i]
        pm2s2=df['sentence2_'+m2+'_prob'][i]
            
        if df['rating'][i]<4:
            
            if sent_pair in res_dict[m1]:
                
                res_dict[m1][sent_pair][1].append(0)
                res_dict[m2][sent_pair][1].append(0)
                
            else:
                
                if pm1s1-pm1s2>0:
                    res_dict[m1][sent_pair]=[1,[0]]
                else:
                    res_dict[m1][sent_pair]=[2,[0]]
                    
                    
                if pm2s1-pm2s2>0:
                    res_dict[m2][sent_pair]=[1,[0]]
                else:
                    res_dict[m2][sent_pair]=[2,[0]]
                    
        elif df['rating'][i]>3:
            
            if sent_pair in res_dict[m1]:
                
                res_dict[m1][sent_pair][1].append(1)
                res_dict[m2][sent_pair][1].append(1)
                
            else:
                
                if pm1s1-pm1s2>0:
                    res_dict[m1][sent_pair]=[1,[1]]
                else:
                    res_dict[m1][sent_pair]=[2,[1]]
                    
                    
                if pm2s1-pm2s2>0:
                    res_dict[m2][sent_pair]=[1,[1]]
                else:
                    res_dict[m2][sent_pair]=[2,[1]]

  



sents_prob_dict= pickle.load( open( "resources/contstim_stim_probs_dict.pkl", "rb" ))
sents_all=list(sents_prob_dict['trigram'].keys())
sents_gcorrs=np.load('resources/contstim_sents_gcorrs.npy')
sents_freqs=np.load('resources/contstim_sents_freqs.npy')














xs=[[] for i in range(9)]
ys=[[] for i in range(9)]
x1s=[[] for i in range(9)]
y1s=[[] for i in range(9)]

cs=[[] for i in range(9)]

for mi,model in enumerate(models):
    
    model_res=res_dict[model]
    
    for sent_pair in model_res.keys():
        
        res=model_res[sent_pair]
        
        s1=sent_pair.split('_')[0]
        s2=sent_pair.split('_')[1]
        
        s1g=sents_gcorrs[sents_all.index(s1)]
        s2g=sents_gcorrs[sents_all.index(s2)]
        
#         s1g=sents_freqs[sents_all.index(s1)]
#         s2g=sents_freqs[sents_all.index(s2)]
        
        if res[0]==1:
            
            
    
            if np.mean(res[1])<.5:
            
                x1s[mi].append(s1g)
                y1s[mi].append(s2g)

                xs[mi].append(s1g)
                ys[mi].append(s2g)
                
                cs[mi].append([0,0,1])
                
            elif np.mean(res[1])>.5:
                
                x1s[mi].append(s2g)
                y1s[mi].append(s1g)
                
                xs[mi].append(s1g)
                ys[mi].append(s2g)
                
                cs[mi].append([1,0,0])
            
        elif res[0]==2:
            
            
            
            if np.mean(res[1])<.5:
                
                x1s[mi].append(s1g)
                y1s[mi].append(s2g)
                
                xs[mi].append(s2g)
                ys[mi].append(s1g)
                
                cs[mi].append([1,0,0])
                
            elif np.mean(res[1])>.5:
                
                x1s[mi].append(s2g)
                y1s[mi].append(s1g)
                
                xs[mi].append(s2g)
                ys[mi].append(s1g)
                
                cs[mi].append([0,0,1])

        


import matplotlib
plt.rcParams['figure.dpi']= 1000
plt.rcParams['savefig.dpi']= 1000


model_labels=['GPT-2','RoBERTa','ELECTRA','BERT','XLM','LSTM','RNN','3-gram','2-gram']

matplotlib.rcParams.update({'font.family':'sans-serif'})
matplotlib.rcParams.update({'font.sans-serif':'Arial'})
# figs, axs = plt.subplots(3,3,aspect='equal')

plt.subplots_adjust(wspace=0, hspace=0)

fig = plt.figure(figsize=(4,4)) # Notice the equal aspect ratio
axs = [fig.add_subplot(3,3,i+1) for i in range(9)]

for a in axs:
#     a.set_xticklabels([])
#     a.set_yticklabels([])
    a.set_aspect('equal')

fig.subplots_adjust(wspace=0, hspace=.1)

ps=[]


for i,ax in enumerate(axs):
    
    
    
    y=i%3
    x=int(np.floor(i/3))
    

#     ax=axs[x,y]
    
    
    ax.set_ylim(0,.7)
    ax.set_xlim(0,.7)

    ax.set_aspect('equal')
    
   
    ax.plot(np.arange(8)/10,np.arange(8)/10,color=[0.7,.7,.7],linewidth=.5)
    
    
    sc3=ax.scatter(x1s[i],y1s[i],s=5,color=[1,0,0],alpha=1)
    
    sc1=ax.scatter([x for xi,x in enumerate(xs[i]) if cs[i][xi]==[0,0,1]],[y for yi,y in enumerate(ys[i]) if cs[i][yi]==[0,0,1]],s=1,color=[0,0,0])
    
    sc2=ax.scatter([x for xi,x in enumerate(xs[i]) if cs[i][xi]==[1,0,0]],[y for yi,y in enumerate(ys[i]) if cs[i][yi]==[1,0,0]],s=1,color=[0,0,0])

    
    confidence_ellipse(xs[i], ys[i], ax, n_std=2,edgecolor='black')
    confidence_ellipse(x1s[i], y1s[i], ax, n_std=2,edgecolor='red')
    
    d1=[x-y for x,y in zip(xs[i],ys[i])]
    d2=[x-y for x,y in zip(x1s[i],y1s[i])]
    
    d3=[m-n for m,n in zip(d1,d2)]
    

    tt=stats.ttest_1samp(d3, popmean=0)
    
    pval=tt.pvalue*9
    
    ps.append(tt.pvalue)
    
    
    ax.tick_params(color=[.7,.7,.7])
    for spine in ax.spines.values():
        spine.set_edgecolor([.7,.7,.7])

    
        
    ax.set_title(model_labels[i],fontsize=6,y=.8,x=0.25)
    
    ax.tick_params(axis='both', which='major', labelsize=4)

    
    if y==0 and x==2:
        ax.set_xlabel('semantic coherence for \n preferred sentence',fontsize=6)
        ax.set_ylabel('semantic coherence for \n rejected sentence',fontsize=6)
    
        ax.text(.27,.58,'preferred low \n coherence',fontsize=6)
        
        ax.text(.56,.19,'preferred high \n coherence',fontsize=6, rotation='vertical')
        
        ax.set_yticks([0,0.2,0.4,0.6])

       
        
    else:
        ax.set_yticks([], [])
        ax.set_xticks([], [])
        
        

        
        
    if y==1 and x==2:

        ax.legend((sc1,sc3),('model preferences', 'human preferences'),fontsize=6,bbox_to_anchor=(1.05, 0.03),frameon=False)
     

    
fdrs=statsmodels.stats.multitest.fdrcorrection(ps, alpha=0.05, method='indep', is_sorted=False)
fdrs=fdrs[1]

for ai,ax in enumerate(axs):
    
    if fdrs[ai]<.05:
        ax.tick_params(color='black')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')

plt.subplots_adjust(wspace=0)

plt.savefig("semantic_coherence.pdf", format="pdf", bbox_inches="tight") 











xs=[[] for i in range(9)]
ys=[[] for i in range(9)]
x1s=[[] for i in range(9)]
y1s=[[] for i in range(9)]

cs=[[] for i in range(9)]

for mi,model in enumerate(models):
    
    model_res=res_dict[model]
    
    for sent_pair in model_res.keys():
        
        res=model_res[sent_pair]
        
        s1=sent_pair.split('_')[0]
        s2=sent_pair.split('_')[1]
        
#         s1g=sents_gcorrs[sents_all.index(s1)]
#         s2g=sents_gcorrs[sents_all.index(s2)]
        
        s1g=sents_freqs[sents_all.index(s1)]
        s2g=sents_freqs[sents_all.index(s2)]
        
        if res[0]==1:
            
            
    
            if np.mean(res[1])<.5:
            
                x1s[mi].append(s1g)
                y1s[mi].append(s2g)

                xs[mi].append(s1g)
                ys[mi].append(s2g)
                
                cs[mi].append([0,0,1])
                
            elif np.mean(res[1])>.5:
                
                x1s[mi].append(s2g)
                y1s[mi].append(s1g)
                
                xs[mi].append(s1g)
                ys[mi].append(s2g)
                
                cs[mi].append([1,0,0])
            
        elif res[0]==2:
            
            
            
            if np.mean(res[1])<.5:
                
                x1s[mi].append(s1g)
                y1s[mi].append(s2g)
                
                xs[mi].append(s2g)
                ys[mi].append(s1g)
                
                cs[mi].append([1,0,0])
                
            elif np.mean(res[1])>.5:
                
                x1s[mi].append(s2g)
                y1s[mi].append(s1g)
                
                xs[mi].append(s2g)
                ys[mi].append(s1g)
                
                cs[mi].append([0,0,1])

        




import matplotlib
import statsmodels
plt.rcParams['figure.dpi']= 1000
plt.rcParams['savefig.dpi']= 1000


model_labels=['GPT-2','RoBERTa','ELECTRA','BERT','XLM','LSTM','RNN','3-gram','2-gram']

matplotlib.rcParams.update({'font.family':'sans-serif'})
matplotlib.rcParams.update({'font.sans-serif':'Arial'})
# figs, axs = plt.subplots(3,3,aspect='equal')

plt.subplots_adjust(wspace=0, hspace=0)

# plt.title('(b) average word frequency')

fig = plt.figure(figsize=(4,4)) # Notice the equal aspect ratio
axs = [fig.add_subplot(3,3,i+1) for i in range(9)]

for a in axs:
#     a.set_xticklabels([])
#     a.set_yticklabels([])
    a.set_aspect('equal')

fig.subplots_adjust(wspace=0, hspace=.1)

ps=[]


for i,ax in enumerate(axs):
    
    
    
    y=i%3
    x=int(np.floor(i/3))
    

#     ax=axs[x,y]
    
    
    ax.set_ylim(1,6)
    ax.set_xlim(1,6)

    ax.set_aspect('equal')
    
    ax.plot(np.arange(2,13)/2,np.arange(2,13)/2,color=[0.7,.7,.7],linewidth=.5)

    
    
    sc3=ax.scatter(x1s[i],y1s[i],s=5,color=[1,0,0],alpha=1)
    
    sc1=ax.scatter([x for xi,x in enumerate(xs[i]) if cs[i][xi]==[0,0,1]],[y for yi,y in enumerate(ys[i]) if cs[i][yi]==[0,0,1]],s=1,color=[0,0,0])
    
    sc2=ax.scatter([x for xi,x in enumerate(xs[i]) if cs[i][xi]==[1,0,0]],[y for yi,y in enumerate(ys[i]) if cs[i][yi]==[1,0,0]],s=1,color=[0,0,0])

    
    confidence_ellipse(xs[i], ys[i], ax, n_std=2,edgecolor='black')
    confidence_ellipse(x1s[i], y1s[i], ax, n_std=2,edgecolor='red')
    
    d1=[x-y for x,y in zip(xs[i],ys[i])]
    d2=[x-y for x,y in zip(x1s[i],y1s[i])]
    
    d3=[m-n for m,n in zip(d1,d2)]
    
#     tt=stats.ttest_ind(d1, d2, equal_var=False)
    
    tt=stats.ttest_1samp(d3, popmean=0)
    
    pval=tt.pvalue*9
    
    ps.append(tt.pvalue)
    
    
    ax.tick_params(color=[.7,.7,.7])
    for spine in ax.spines.values():
        spine.set_edgecolor([.7,.7,.7])

    
        
    ax.set_title(model_labels[i],fontsize=6,y=.8,x=0.25)
    
#     if i!=2:
#     ax.plot(np.arange(8)/10,np.arange(8)/10,color=[0,0,0],linewidth=.5,alpha=.5)
    

    ax.tick_params(axis='both', which='major', labelsize=4)

    
    if y==0 and x==2:
        ax.set_xlabel('avg –log word frequency \n for preferred sentence',fontsize=6)
        ax.set_ylabel('avg –log word frequency \n for rejected sentence',fontsize=6)
    
        ax.text(2.1,1.3,'preferred high \n frequency',fontsize=6)
        
        ax.text(1.2,2.1,'preferred low \n frequency',fontsize=6, rotation='vertical')

       
        
    else:
        ax.set_yticks([], [])
        ax.set_xticks([], [])
        
        

        
        
    if y==1 and x==2:

        ax.legend((sc1,sc3),('model preferences', 'human preferences'),fontsize=6,bbox_to_anchor=(1.05, 0.03),frameon=False)
     

    
fdrs=statsmodels.stats.multitest.fdrcorrection(ps, alpha=0.05, method='indep', is_sorted=False)
fdrs=fdrs[1]

for ai,ax in enumerate(axs):
    
    if fdrs[ai]<.05:
        ax.tick_params(color='black')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')

plt.subplots_adjust(wspace=0)

plt.savefig("frequency.pdf", format="pdf", bbox_inches="tight") 


