import pickle
import dill
import numpy as np
import torch
from collections import defaultdict
from itertools import count
import torch.nn as nn

########################################################
    
with open('vocab_low.pkl', 'rb') as file:
    vocab_low=pickle.load(file) 
    
with open('vocab_low_freqs.pkl', 'rb') as file:
    vocab_low_freqs=pickle.load(file) 

with open('vocab_cap.pkl', 'rb') as file:
    vocab_cap=pickle.load(file) 
    
with open('vocab_cap_freqs.pkl', 'rb') as file:
    vocab_cap_freqs=pickle.load(file) 
    
########################################################   
    
with open('neuralnet_word2id_dict.pkl', 'rb') as file:
    word2id=pickle.load(file)
    
id2word=dict(zip([word2id[w] for w in word2id],[w for w in word2id]))

########################################################   

# Hyper-parameters
embed_size = 256
hidden_size = 512
num_layers = 1
batch_size = 1

vocab_size=np.max([word2id[w] for w in word2id])+1



# model = RNNModel(embed_size, hidden_size, num_layers, vocab_size).to(device)

########################################################   

model=torch.load('contstim_rnn.pt')
model=model.to('cuda')

########################################################  

def rnn_sent_prob(sent):   

    prompt='This is a sentence to calibrate the internal state'

    plen=len(prompt.split())
    
    states = (torch.zeros(num_layers, batch_size, hidden_size).to('cuda'),
                  torch.zeros(num_layers, batch_size, hidden_size).to('cuda'))

    words=prompt.split() + ['.'] + sent.split() + ['.']

    inputs = torch.tensor([word2id[w] for w in words]).to('cuda').unsqueeze(0)
    
    h0 = torch.zeros(num_layers, batch_size, hidden_size).to('cuda')

    outputs, states = model(inputs, h0)

    soft=torch.softmax(outputs,-1).cpu().data.numpy()[0]

    prob=float(np.prod([float(soft[wi+plen,word2id[w]]) for wi,w in enumerate(words[1+plen:])]))

    return prob

########################################################  

def rnn_word_probs(words,wordi):

    if wordi>0:
        vocab=vocab_low
    else:
        vocab=vocab_cap


    prompt='This is a sentence to calibrate the internal state of the model'

    plen=len(prompt.split())

    wordi=wordi+1+plen

    words=prompt.split() + ['.'] + words + ['.']

    h0 = torch.zeros(num_layers, batch_size, hidden_size).to('cuda')


    inputs = torch.tensor([word2id[w] for w in words]).to('cuda').unsqueeze(0)
    outputs, states = model(inputs, h0)
    soft=torch.softmax(outputs,-1).cpu().data.numpy()[0]

    ss = np.argsort(soft[wordi-1])[::-1]
    top_words=[id2word[s] for s in ss[:3000]]
    top_words=list(set(top_words)&set(vocab))
    inds=[vocab.index(t) for t in top_words]


    probs=[]

    for wi,w in enumerate(top_words):

        states = (torch.zeros(num_layers, batch_size, hidden_size).to('cuda'),
                          torch.zeros(num_layers, batch_size, hidden_size).to('cuda'))

        words[wordi]=w

        inputs = torch.tensor([word2id[w] for w in words]).to('cuda').unsqueeze(0)

        outputs, states = model(inputs, h0)


        soft=torch.softmax(outputs,-1).cpu().data.numpy()[0]

        prob=float(np.prod([float(soft[wordi-1+x,word2id[w1]]) for x,w1 in enumerate(words[wordi:])]))

        probs.append(prob)

    probs=np.array(probs)

    return probs, inds

