#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import random
import math

from lstm_class import RNNLM
from bilstm_class import RNNLM_bilstm
from rnn_class import RNNModel
from model_functions import model_factory

models=['bigram','trigram','rnn','lstm','bilstm','bert','bert_whole_word','roberta','xlm','electra','gpt2']

#load probability ranges and get steps
rngs=np.load('prob_ranges.npy')
steps_all=np.zeros([11,11])
for i,rng in enumerate(rngs):
    low=rng[0]-45
    high=rng[1]
    steps=np.linspace(low,high,11)
    steps_all[i,:]=steps

#turn on/off printing (1=on)
print_on=1

#model names
model1_name='gpt2'

#bigram and trigram models run on CPU, so gpu_id will be ignored
model1_gpu_id=0

#get probability range for model 1
steps=steps_all[models.index(model1_name)]

#sentence length
sent_len=8


# In[ ]:


model1=model_factory(model1_name,model1_gpu_id)

get_model1_sent_prob=model1.sent_prob
get_model1_word_probs=model1.word_probs


# In[ ]:


with open('vocab_low.pkl', 'rb') as file:
    vocab_low=pickle.load(file) 
    
with open('vocab_low_freqs.pkl', 'rb') as file:
    vocab_low_freqs=pickle.load(file) 

with open('vocab_cap.pkl', 'rb') as file:
    vocab_cap=pickle.load(file) 
    
with open('vocab_cap_freqs.pkl', 'rb') as file:
    vocab_cap_freqs=pickle.load(file) 


# In[ ]:


for step_ind in range(10):
    
    step_low=steps[step_ind]
    step_high=steps[step_ind+1]
    
    file=open(model1_name+'_level_'+str(step_ind+1)+'.txt','w')
    
    num_sents=0
    while num_sents<60:
    
        wordi=np.arange(sent_len)
        wordis=[]  
        for i in range(1000):
            random.shuffle(wordi)
            wordis=wordis+list(wordi)

        vocab_low_freqs1=np.ones([len(vocab_low_freqs)])/len(vocab_low_freqs)
        vocab_cap_freqs1=np.ones([len(vocab_cap_freqs)])/len(vocab_cap_freqs)

        words1=list(np.random.choice(vocab_cap, 1, p=vocab_cap_freqs1)) + list(np.random.choice(vocab_low, sent_len-1, p=vocab_low_freqs1, replace=False))

        words1o=words1.copy()

        sent1=' '.join(words1)
        
        sent1_last=sent1

        model1_sent1_prob=get_model1_sent_prob(sent1)

        if print_on==1:
            print(model1_sent1_prob)
            print(sent1)
            print('\n')

        probs_all_list=[model1_sent1_prob]

        cycle=0
        
        for samp in range(10000):                     
            
            if step_ind<9 and np.log(model1_sent1_prob) > step_low and np.log(model1_sent1_prob) < step_high:

                file.write(sent1+'.')
                file.write('\n')
                
                num_sents+=1
                
                break
                
            elif step_ind==9 and cycle==8 and np.log(model1_sent1_prob) > step_low and np.log(model1_sent1_prob) < step_high:

                file.write(sent1+'.')
                file.write('\n')
                
                num_sents+=1
                
                break   
                               
            elif np.log(model1_sent1_prob) > step_high:
                break
                                
            elif np.log(model1_sent1_prob) < step_low and cycle==sent_len:
                break
                                
            
            if samp%sent_len==0:
                cycle=0

            words1o=words1.copy()

            wordi=int(wordis[samp])

            cur_word1=words1[wordi]

            if wordi==0:
                vocab=vocab_cap
            else:
                vocab=vocab_low

            model1_word1_probs = get_model1_word_probs(words1,wordi)
            
            

            if len(model1_word1_probs)==2:
                model1_word1_inds=model1_word1_probs[1]
                model1_word1_probs=model1_word1_probs[0]
            else:
                model1_word1_inds=np.arange(len(vocab))
                
            

            words1=words1o.copy()

            word1_list=[vocab[w] for w in model1_word1_inds]

            model1_word1_probs=model1_word1_probs/np.sum(model1_word1_probs)
        
            word1_tops=[word1_list[vp] for vp in np.argsort(model1_word1_probs)[::-1][:10]] + [cur_word1]

            model1_sent1_probs=[]

            sent1_conts12=[]
            sent2_conts21=[]

            for word1 in word1_tops:

                words1t=words1o.copy()

                words1t[wordi]=word1

                sent1t=' '.join(words1t)

                model1_sent1t_prob=get_model1_sent_prob(sent1t)

                model1_sent1_probs.append(model1_sent1t_prob)

            aa=np.argmax(model1_sent1_probs)

            new_word1=word1_tops[aa]

            new_word1o=new_word1

            words1[wordi]=new_word1.upper()

            sent1p=' '.join(words1)

            words1[wordi]=new_word1.lower()
            if wordi==0 or new_word1o[0].isupper():
                words1[wordi]=new_word1.lower().capitalize()

            sent1=' '.join(words1)
            
            if sent1==sent1_last:
                cycle+=1

            sent1_last=sent1

            model1_sent1_prob=get_model1_sent_prob(sent1)

            probs_all_list.append(model1_sent1_prob)

            if print_on==1:
                print(model1_sent1_prob)
                print(sent1p)
                print('\n')

            
            
                
            
            

            
            
            
            




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




