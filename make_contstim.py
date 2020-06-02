#!/usr/bin/env python
# coding: utf-8

# In[1]:


#list of model names
#gpt2
#bert
#bert_whole_word
#roberta
#electra
#xlm
#lstm
#rnn

squash_threshold=100 
model1='electra'
model2='lstm'


# In[2]:


import pickle
import numpy as np
import random
import math

from lstm_class import RNNLM
from rnn_class import RNNModel

exec('from ' +model1+'_functions import *')
exec('from ' +model2+'_functions import *')

exec('get_model1_sent_prob=' +model1+ '_sent_prob')
exec('get_model1_word_probs=' +model1+ '_word_probs')
exec('get_model2_sent_prob=' +model2+ '_sent_prob')
exec('get_model2_word_probs=' +model2+ '_word_probs')


# In[3]:



with open('vocab_low.pkl', 'rb') as file:
    vocab_low=pickle.load(file) 
    
with open('vocab_low_freqs.pkl', 'rb') as file:
    vocab_low_freqs=pickle.load(file) 

with open('vocab_cap.pkl', 'rb') as file:
    vocab_cap=pickle.load(file) 
    
with open('vocab_cap_freqs.pkl', 'rb') as file:
    vocab_cap_freqs=pickle.load(file) 


# In[4]:


def squash(prob,squash_threshold):
    prob=10*np.log(1+math.e**((prob+squash_threshold)/10))-squash_threshold
    return prob


def cont_score(model1_sent1_prob,model1_sent2_prob,model2_sent1_prob,model2_sent2_prob):
    
    gamma = 100 # subject noise

    model1_sent1_prob=squash(np.log(model1_sent1_prob),squash_threshold)
    model1_sent2_prob=squash(np.log(model1_sent2_prob),squash_threshold)
    model2_sent1_prob=squash(np.log(model2_sent1_prob),squash_threshold)
    model2_sent2_prob=squash(np.log(model2_sent2_prob),squash_threshold)
    
    s1a_b = model1_sent1_prob - model1_sent2_prob
    s2a_b = model2_sent1_prob - model2_sent2_prob
            

    # Prob that model 1 is better and a subject picks a/b
    p1a = (1/2) / (1 + np.exp(-(s1a_b)/gamma))
    p1b = 1/2 - p1a

    # Prob that model 2 is better and a subject picks a/b
    p2a = (1/2) / (1 + np.exp(-(s2a_b)/gamma))
    p2b = 1/2 - p2a


    # Mutual information of model and sentence pick
    # Each term is p(model,sent)*log(p(model,sent)/(p(model)*p(sent)))
    conta = p1a * np.log2(p1a/((p1a+p2a)/2)) +              p2a * np.log2(p2a/((p1a+p2a)/2)) 
    
    contb = p1b * np.log2(p1b/((p1b+p2b)/2)) +              p2b * np.log2(p2b/((p1b+p2b)/2))
    
    return conta,contb


# In[ ]:


sent_len=8

wordi=np.arange(sent_len)
wordis=[]  
for i in range(1000):
    random.shuffle(wordi)
    wordis=wordis+list(wordi)
     
words1=list(np.random.choice(vocab_cap, 1, p=vocab_cap_freqs)) + list(np.random.choice(vocab_low, sent_len-1, p=vocab_low_freqs, replace=False))
#words2=list(np.random.choice(vocab_cap, 1, p=vocab_cap_freqs)) + list(np.random.choice(vocab_low, sent_len-1, p=vocab_low_freqs, replace=False))
words2=words1.copy()

# wordis=np.load('wordis.npy')

# words1='Can be but the I leave justice hold'.split()
# words2='Forget an it talking a take my guys'.split()

# words1='Love you missing after make knew grow your'.split()
# words2='Love you missing after make knew grow your'.split()

# words1='Eric need nothing exchange honey truth past us'.split()
# words2='Eric need nothing exchange honey truth past us'.split()


words1o=words1.copy()
words2o=words2.copy()

sent1=' '.join(words1)
sent2=' '.join(words2)


model1_sent1_prob=get_model1_sent_prob(sent1)
model2_sent1_prob=get_model2_sent_prob(sent1)

model1_sent2_prob=get_model1_sent_prob(sent2)
model2_sent2_prob=get_model2_sent_prob(sent2)


conta_last,contb_last=cont_score(model1_sent1_prob,model1_sent2_prob,model2_sent1_prob,model2_sent2_prob) 


prob_diff_last12=np.log(model1_sent1_prob/model2_sent1_prob)
prob_diff_last21=np.log(model2_sent2_prob/model1_sent2_prob)

probs_all=[model1_sent1_prob,model2_sent1_prob,model1_sent2_prob,model2_sent2_prob]

probs_all_list=[]
print(probs_all)
print(sent1)
print(sent2)
print('\n')
    


for samp in range(10000):
    
    words1o=words1.copy()
    words2o=words2.copy()

    wordi=int(wordis[samp])
    
    cur_word1=words1[wordi]
    cur_word2=words2[wordi]
    
    if wordi==0:
        vocab=vocab_cap
    else:
        vocab=vocab_low
    
    
    model1_word1_probs = get_model1_word_probs(words1,wordi)
    model1_word2_probs = get_model1_word_probs(words2,wordi)

    
    if len(model1_word1_probs)==2:
        model1_word1_inds=model1_word1_probs[1]
        model1_word1_probs=model1_word1_probs[0]
        model1_word2_inds=model1_word2_probs[1]
        model1_word2_probs=model1_word2_probs[0]
    else:
        model1_word1_inds=np.arange(len(vocab))
        model1_word2_inds=np.arange(len(vocab))
          
    words1=words1o.copy()
    words2=words2o.copy()
    
    model2_word1_probs = get_model2_word_probs(words1,wordi)
    model2_word2_probs = get_model2_word_probs(words2,wordi)
    

    if len(model2_word1_probs)==2:
        model2_word1_inds=model2_word1_probs[1]
        model2_word1_probs=model2_word1_probs[0]
        model2_word2_inds=model2_word2_probs[1]
        model2_word2_probs=model2_word2_probs[0]
    else:
        model2_word1_inds=np.arange(len(vocab))
        model2_word2_inds=np.arange(len(vocab))

    word1_inds=list(set(model1_word1_inds)&set(model2_word1_inds))
    word2_inds=list(set(model1_word2_inds)&set(model2_word2_inds))
    
#     sys.e
    

    model1_word1_probs=[model1_word1_probs[wi] for wi,i in enumerate(word1_inds) if i in model1_word1_inds]
    model1_word2_probs=[model1_word2_probs[wi] for wi,i in enumerate(word2_inds) if i in model1_word2_inds]
    model2_word1_probs=[model2_word1_probs[wi] for wi,i in enumerate(word1_inds) if i in model2_word1_inds]
    model2_word2_probs=[model2_word2_probs[wi] for wi,i in enumerate(word2_inds) if i in model2_word2_inds]
    
    
    word1_list=[vocab[w] for w in word1_inds]
    word2_list=[vocab[w] for w in word2_inds]
    
    
        
    model1_word1_probs=model1_word1_probs/np.sum(model1_word1_probs)
    model1_word2_probs=model1_word2_probs/np.sum(model1_word2_probs)
    
    model2_word1_probs=model2_word1_probs/np.sum(model2_word1_probs)
    model2_word2_probs=model2_word2_probs/np.sum(model2_word2_probs)
    
    word1_probs12=model1_word1_probs*np.log(model1_word1_probs/model2_word1_probs)
    word2_probs21=model2_word2_probs*np.log(model2_word2_probs/model1_word2_probs)

    word1_probs12[np.isnan(word1_probs12)]=0
    word1_probs12[np.where(word1_probs12<0)[0]]=0
    word1_probs12=word1_probs12/np.sum(word1_probs12)
    
    word2_probs21[np.isnan(word2_probs21)]=0
    word2_probs21[np.where(word2_probs21<0)[0]]=0
    word2_probs21=word2_probs21/np.sum(word2_probs21)

    word1_tops=[word1_list[vp] for vp in np.argsort(word1_probs12)[::-1][:10]] + [cur_word1]
    word2_tops=[word2_list[vp] for vp in np.argsort(word2_probs21)[::-1][:10]] + [cur_word2]

    model1_sent1_probs=[]
    model1_sent2_probs=[]
    model2_sent1_probs=[]
    model2_sent2_probs=[]
    
    sent1_conts12=[]
    sent2_conts21=[]
    
    for word1,word2 in zip(word1_tops,word2_tops):
        
        words1t=words1o.copy()
        words2t=words2o.copy()
        
        words1t[wordi]=word1
        words2t[wordi]=word2
        
        sent1t=' '.join(words1t)
        sent2t=' '.join(words2t)

        model1_sent1t_prob=get_model1_sent_prob(sent1t)
        model2_sent1t_prob=get_model2_sent_prob(sent1t)
        
        model1_sent2t_prob=get_model1_sent_prob(sent2t)
        model2_sent2t_prob=get_model2_sent_prob(sent2t)
        
        model1_sent1_probs.append(model1_sent1t_prob)
        model1_sent2_probs.append(model1_sent2t_prob)
    
        model2_sent1_probs.append(model2_sent1t_prob)
        model2_sent2_probs.append(model2_sent2t_prob)
        

        
        
    conta_scores=[]
    contb_scores=[]
    cont_score_inds=[]
    a=-1  
    for m11,m21 in zip(model1_sent1_probs,model2_sent1_probs):
        a+=1
        b=-1
        for m12,m22 in zip(model1_sent2_probs,model2_sent2_probs):
            b+=1
            
            conta,contb = cont_score(m11,m12,m21,m22) 
            conta_scores.append(conta)
            contb_scores.append(contb)
            cont_score_inds.append([a,b])
  
    aa=cont_score_inds[np.argmax(conta_scores)][0]
    bb=cont_score_inds[np.argmax(contb_scores)][1]
    
    
    
    
    new_word1=word1_tops[aa]
    new_word2=word2_tops[bb]

    new_word1o=new_word1
    new_word2o=new_word2

    words1[wordi]=new_word1.upper()
    words2[wordi]=new_word2.upper()
    
    sent1p=' '.join(words1)
    sent2p=' '.join(words2)

    words1[wordi]=new_word1.lower()
    if wordi==0 or new_word1o[0].isupper():
        words1[wordi]=new_word1.lower().capitalize()

    words2[wordi]=new_word2.lower()
    if wordi==0 or new_word2o[0].isupper():
        words2[wordi]=new_word2.lower().capitalize()
    
    sent1=' '.join(words1)
    sent2=' '.join(words2)

    model1_sent1_prob=get_model1_sent_prob(sent1)
    model2_sent1_prob=get_model2_sent_prob(sent1)

    model1_sent2_prob=get_model1_sent_prob(sent2)
    model2_sent2_prob=get_model2_sent_prob(sent2)


    probs_all=[model1_sent1_prob,model2_sent1_prob,model1_sent2_prob,model2_sent2_prob]
    
    probs_all_list.append(probs_all)

    print(probs_all)
    print(sent1p)
    print(sent2p)
    print('\n')




    


# In[ ]:




