from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import torch
import math
import itertools
import random
import pickle

######################################################################

with open('vocab_low.pkl', 'rb') as file:
    vocab_low=pickle.load(file) 
    
with open('vocab_low_freqs.pkl', 'rb') as file:
    vocab_low_freqs=pickle.load(file) 

with open('vocab_cap.pkl', 'rb') as file:
    vocab_cap=pickle.load(file) 
    
with open('vocab_cap_freqs.pkl', 'rb') as file:
    vocab_cap_freqs=pickle.load(file) 
    
######################################################################
    
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
model=model.to('cuda')

#add pad token
special_tokens_dict = {'pad_token': '[PAD]'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))  

######################################################################

starts_gpt2=[]
suffs_gpt2=[]

for i in range(len(tokenizer.get_vocab())):
    tok=tokenizer.decode(i)
    if tok[0]==' ' or tok[0]=='.' or tok=='[MASK]':
        starts_gpt2.append(i)
    elif tok[0]!=' ':
        suffs_gpt2.append(i)
        
starts_gpt2_cuda=torch.tensor(starts_gpt2).to('cuda')
suffs_gpt2_cuda=torch.tensor(suffs_gpt2).to('cuda')

######################################################################

toklist_low=[]
for v in vocab_low:
    toks=tokenizer.encode(' '+v)
    toklist_low.append(toks)
    
toklist_cap=[]
for v in vocab_cap:
    toks=tokenizer.encode(' '+v)
    toklist_cap.append(toks) 
    
######################################################################


def gpt2_word_probs(words,wordi):
    
    if wordi==0:
        vocab=vocab_cap
        toklist=toklist_cap
    else:
        vocab=vocab_low
        toklist=toklist_low

    
    sent1=' '.join(words[:wordi])
    sent2=' '.join(words[wordi+1:])

    tok1=tokenizer.encode('. ' + sent1)
    tok2=tokenizer.encode(' ' + sent2)

        
    ####################################################3## 
        
    lp=0
    while 0==0:
        in1=tok1
        in1=torch.tensor(in1).to('cuda')

        with torch.no_grad():
            out1=model(in1)[0]
            soft1=torch.softmax(out1,-1)[-1].cpu().data.numpy()

        logsoft1=np.log(soft1)

        tops=np.where(logsoft1>-10-lp*5)[0]

        tops=[t for t in tops if t in starts_gpt2]
        
        if len(tops)<10:
            lp=lp+1
        else:
            break

    ##########################
        
    inputs=[]
    vocab_to_input_inds=[]
    vocab_to_input_pred_vocs=[]
    vocab_to_input_pos=[]
    
    vocab_tops=[]
    vocab_tops_ind=[]
    

    for wi,word in enumerate(vocab):
        
        wordtok=toklist[wi]
        
        if wordtok[0] in tops:
            
            vocab_tops.append(word)
            vocab_tops_ind.append(wi)

            in1 = tok1 + wordtok + tok2 + tokenizer.encode('.')

            inputs.append(in1)
        
   
    maxlen=np.max([len(i) for i in inputs])

    inputs0=[i+[0]*(maxlen-len(i)) for i in inputs]
    att_mask=np.ceil(np.array(inputs0)/100000)
    
    inputs=[i+[tokenizer.pad_token_id]*(maxlen-len(i)) for i in inputs]
    
    batchsize=200

    for i in range(int(np.ceil(len(inputs)/batchsize))):

        inputs1=np.array(inputs[batchsize*i:batchsize*(i+1)])
        
        att_mask1=att_mask[batchsize*i:batchsize*(i+1)]

        inputs2=torch.tensor(inputs1).to('cuda')
        att_mask1=torch.tensor(att_mask1).to('cuda')
       
        
        with torch.no_grad():
            
            out1=model(input_ids=inputs2,attention_mask=att_mask1)[0]           

            out_suff_inds=torch.where(torch.tensor(np.in1d(inputs1,suffs_gpt2).reshape(inputs1.shape[0],-1)).to('cuda')==True)  
            
            out_start_inds=torch.where(torch.tensor(np.in1d(inputs1,starts_gpt2).reshape(inputs1.shape[0],-1)).to('cuda')==True) 

#             for x in range(len(out_suff_inds[0])):
#                 out1[out_suff_inds[0][x],out_suff_inds[1][x]-1,starts_gpt2_cuda]=math.inf*-1

#             for x in range(len(out_start_inds[0])):
#                 out1[out_start_inds[0][x],out_start_inds[1][x]-1,suffs_gpt2_cuda]=math.inf*-1

            soft=torch.softmax(out1,-1)
            
       
            for v in range(len(inputs1)):
    
                numwords=len(np.where(inputs1[v]<tokenizer.pad_token_id)[0])-1

                probs=torch.tensor([soft[v,n,inputs1[v][n+1]] for n in range(len(tok1)-1,numwords)])

                prob=torch.prod(probs)#.cpu().data.numpy())

                if i==0 and v==0:
                    vocab_probs=prob.unsqueeze(0)
                else:
                    vocab_probs=torch.cat((vocab_probs,prob.unsqueeze(0)),0)
                    
                    
    vocab_probs=vocab_probs.cpu().data.numpy()
    
    return vocab_probs, vocab_tops_ind
                   
                    
def gpt2_sent_prob(sent):
    
    sent='. ' + sent + '.'
    
    tokens=tokenizer.encode(sent)
    inputs=torch.tensor(tokens).to('cuda')
    
    with torch.no_grad():
        out=model(inputs)

    unsoft=out[0]
    lab1=inputs.cpu().data.numpy()
    
    probs=[]
    for x in range(len(lab1)-1):
        
        lab=lab1[x+1]       
        unsoft1=unsoft[x]
        
        if lab in starts_gpt2:
            
            soft=torch.softmax(unsoft1[starts_gpt2],-1)            
            prob=float(soft[starts_gpt2.index(lab)].cpu().data.numpy())
                      
        elif lab in suffs_gpt2:
            
            soft=torch.softmax(unsoft1[suffs_gpt2],-1)          
            prob=float(soft[suffs_gpt2.index(lab)].cpu().data.numpy())
            
        probs.append(prob)
        
    prob=np.prod(probs)
    
    return prob


######################################################################
