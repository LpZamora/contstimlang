from transformers import BertForMaskedLM, BertTokenizer
import numpy as np
import torch
import math
import itertools
import random
import pickle

###################################

with open('vocab_low.pkl', 'rb') as file:
    vocab_low=pickle.load(file) 
    
with open('vocab_low_freqs.pkl', 'rb') as file:
    vocab_low_freqs=pickle.load(file) 

with open('vocab_cap.pkl', 'rb') as file:
    vocab_cap=pickle.load(file) 
    
with open('vocab_cap_freqs.pkl', 'rb') as file:
    vocab_cap_freqs=pickle.load(file) 

######################################################################

tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertForMaskedLM.from_pretrained('bert-large-cased-whole-word-masking')
model=model.to('cuda')

######################################################################

starts=[]
suffs=[]

for i in range(len(tokenizer.get_vocab())):
    tok=tokenizer.decode(i)
    if tok[0]!='#':
        starts.append(i)
    elif tok[0]!=' ':
        suffs.append(i)
        
######################################################################

toklist_low=[]
for vi,v in enumerate(vocab_low):
    toks=tokenizer.encode(v)[1:-1]
    toklist_low.append(toks)
    
toklist_cap=[]
for vi,v in enumerate(vocab_cap):
    toks=tokenizer.encode(v)[1:-1]
    toklist_cap.append(toks)

    
tokparts_all_low=[]
tps_low=[]
for ti,tokens in enumerate(toklist_low):

    tokparts_all=[]
    tokens=[tokenizer.mask_token_id]*len(tokens)
    tok_perms=list(itertools.permutations(np.arange(len(tokens)),len(tokens)))    
       
    for perms in tok_perms:    
        
        tokparts=[tokens]      
        tokpart=[tokenizer.mask_token_id]*len(tokens)        
        tps_low.append(tokpart.copy())

        for perm in perms[:-1]:
            
            tokpart[perm]=tokens[perm]            
            tokparts.append(tokpart.copy())            
            tps_low.append(tokpart.copy()) 
            
        tokparts_all.append(tokparts)
    
    tokparts_all_low.append(tokparts_all)
        

        
tokparts_all_cap=[]
tps_cap=[]
for ti,tokens in enumerate(toklist_cap):

    tokparts_all=[]    
    tokens=[tokenizer.mask_token_id]*len(tokens)
    tok_perms=list(itertools.permutations(np.arange(len(tokens)),len(tokens)))    
       
    for perms in tok_perms:
        
        tokparts=[tokens]       
        tokpart=[tokenizer.mask_token_id]*len(tokens)        
        tps_cap.append(tokpart.copy())

        for perm in perms[:-1]:
            
            tokpart[perm]=tokens[perm]            
            tokparts.append(tokpart.copy())            
            tps_cap.append(tokpart.copy()) 
            
        tokparts_all.append(tokparts)
    
    tokparts_all_cap.append(tokparts_all)    
    
    
    
######################################################################    
    
batchsize=1000
    
unique_tokparts_low=[list(x) for x in set(tuple(x) for x in tps_low)]

tokparts_inds_low=[]

vocab_probs_sheet_low=[]

vocab_to_tokparts_inds_low=[]

vocab_to_tokparts_inds_map_low=[[] for i in range(int(np.ceil(len(unique_tokparts_low)/batchsize)))]

for vocind,tokparts_all in enumerate(tokparts_all_low):

    inds_low_all=[]
    voc_low_all=[]
    voc_to_inds_low_all=[]
    
    toks=toklist_low[vocind]
    
    tok_perms=list(itertools.permutations(np.arange(len(toks)),len(toks)))   
    
    for ti_all,tokparts in enumerate(tokparts_all):
        
        tok_perm=tok_perms[ti_all]
        
        inds_low=[]
        voc_low=[]
        vocab_to_inds_low=[]
        
        for ti,tokpart in enumerate(tokparts):
            
            tokind=tok_perm[ti]

            ind=unique_tokparts_low.index(tokpart)

            inds_low.append([ind,tokind,toks[tokind]])
            
            voc_low.append(0)
            
            vocab_to_inds_low.append([ind,ti_all,ti])
            
            batchnum=int(np.floor(ind/batchsize))
            
            unique_ind_batch=ind%batchsize
            
            vocab_to_tokparts_inds_map_low[batchnum].append([[vocind,ti_all,ti],[unique_ind_batch,tokind,toks[tokind]]])
        
        inds_low_all.append(inds_low)
        voc_low_all.append(voc_low)
        
    tokparts_inds_low.append(inds_low_all)
    vocab_probs_sheet_low.append(voc_low_all)
    
    vocab_to_tokparts_inds_low.append(vocab_to_inds_low)
    
    
######################################################################

unique_tokparts_cap=[list(x) for x in set(tuple(x) for x in tps_cap)]

tokparts_inds_cap=[]

vocab_probs_sheet_cap=[]

vocab_to_tokparts_inds_cap=[]

vocab_to_tokparts_inds_map_cap=[[] for i in range(int(np.ceil(len(unique_tokparts_cap)/batchsize)))]

for vocind,tokparts_all in enumerate(tokparts_all_cap):

    inds_cap_all=[]
    voc_cap_all=[]
    voc_to_inds_cap_all=[]
    
    toks=toklist_cap[vocind]
    
    tok_perms=list(itertools.permutations(np.arange(len(toks)),len(toks)))   
    
    for ti_all,tokparts in enumerate(tokparts_all):
        
        tok_perm=tok_perms[ti_all]
        
        inds_cap=[]
        voc_cap=[]
        vocab_to_inds_cap=[]
        
        for ti,tokpart in enumerate(tokparts):
            
            tokind=tok_perm[ti]

            ind=unique_tokparts_cap.index(tokpart)

            inds_cap.append([ind,tokind,toks[tokind]])
            
            voc_cap.append(0)
            
            vocab_to_inds_cap.append([ind,ti_all,ti])
            
            batchnum=int(np.floor(ind/batchsize))
            
            unique_ind_batch=ind%batchsize
            
            vocab_to_tokparts_inds_map_cap[batchnum].append([[vocind,ti_all,ti],[unique_ind_batch,tokind,toks[tokind]]])
        
        inds_cap_all.append(inds_cap)
        voc_cap_all.append(voc_cap)
        
    tokparts_inds_cap.append(inds_cap_all)
    vocab_probs_sheet_cap.append(voc_cap_all)
    
    vocab_to_tokparts_inds_cap.append(vocab_to_inds_cap)
    
    

    
    
def bert_whole_word_word_probs(words,wordi):
    
    if wordi>0:
        vocab=vocab_low
        unique_tokparts=unique_tokparts_low
        vocab_probs_sheet=vocab_probs_sheet_low
        vocab_to_tokparts_inds_map=vocab_to_tokparts_inds_map_low
    else:
        vocab=vocab_cap
        unique_tokparts=unique_tokparts_cap
        vocab_probs_sheet=vocab_probs_sheet_cap
        vocab_to_tokparts_inds_map=vocab_to_tokparts_inds_map_cap
    
    
    words1=words.copy()
    words2=words.copy()
    
    words[wordi]=tokenizer.mask_token

    sent=' '.join(words)

    tokens=tokenizer.encode(sent+'.')
    
    mask_ind=tokens.index(tokenizer.mask_token_id)

    tok1=tokens[:mask_ind]
    tok2=tokens[mask_ind+1:]      

    inputs=[]
    for un in unique_tokparts:

        in1=tok1 + un + tok2
        inputs.append(in1)

        
    maxlen=np.max([len(i) for i in inputs])

    inputs=[i+[0]*(maxlen-len(i)) for i in inputs]

    att_mask=np.ceil(np.array(inputs)/100000)

    inputs=torch.tensor(inputs).to('cuda')
    att_mask=torch.tensor(att_mask).to('cuda')

    batchsize=1000

    for i in range(int(np.ceil(len(inputs)/batchsize))):
                
        vocab_to_tokparts_inds_map_batch=vocab_to_tokparts_inds_map[i]

        inputs1=inputs[batchsize*i:batchsize*(i+1)]
        
        att_mask1=att_mask[batchsize*i:batchsize*(i+1)]
        
        mask_inds=[torch.where(inp==tokenizer.mask_token_id)[0]-wordi-1 for inp in inputs1]

        with torch.no_grad():
            
            out1=model(inputs1,attention_mask=att_mask1)[0]

            out1=out1[:,wordi+1:wordi+7,:]
            
#             out1[:,0,suffs]=math.inf*-1               
#             out1[:,1:,starts]=math.inf*-1

            soft=torch.softmax(out1,-1)
            
            for vti in vocab_to_tokparts_inds_map_batch:
                
                vocab_probs_sheet[vti[0][0]][vti[0][1]][vti[0][2]]=float(soft[vti[1][0],vti[1][1],vti[1][2]])

            del soft

    vocab_probs=[]
    for x in range(len(vocab_probs_sheet)):
                
        probs=[]
        for y in range(len(vocab_probs_sheet[x])):

            prob=np.prod(vocab_probs_sheet[x][y])

            probs.append(prob)

        vocab_probs.append(float(np.exp(float(np.mean(np.log(probs))))))

    vocab_probs=np.array(vocab_probs)
    
    return vocab_probs




def bert_whole_word_sent_prob(sent):
    
 
    word_tokens_per=tokenizer.encode(sent+'.')
    word_tokens_per[-2]=tokenizer.mask_token_id
    in1=torch.tensor(word_tokens_per).to('cuda').unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=in1)[0]#.cpu().data.numpy()
        out=out[:,-2,:]   
        out[:,suffs]=math.inf*-1   
        soft=torch.softmax(out,-1).cpu().data.numpy()
    per_cent=soft[0,tokenizer.encode('.')[1:-1]]

                  
    
    words=sent.split(' ')
    
    word_tokens=tokenizer.encode(sent)[1:-1]

    tokens=tokenizer.encode(sent+'.', add_special_tokens=True) 
    
    start_inds=np.where(np.in1d(tokens,starts)==True)[0][:-2]
    suff_inds=np.where(np.in1d(tokens,suffs)==True)[0]

    wordtoks=[tokenizer.encode(w)[1:-1] for w in words]
    senttoks=[item for sublist in wordtoks for item in sublist]

    tokens_all=[]
    labels_all=[]

    tokens_all_inds=[]

    input_to_mask_inds=dict()

    word_inds=list(np.linspace(1,len(words),len(words)).astype('int'))

    msk_inds_all=[]

    for i in range(1,len(words)+1):       
        msk_inds=list(itertools.combinations(word_inds,i))     
        msk_inds=[list(m) for m in msk_inds] 
        msk_inds_all=msk_inds_all+msk_inds

    msk_inds_all=msk_inds_all[::-1]

    for mski,msk_inds in enumerate(msk_inds_all):

        msk_inds=list(np.array(msk_inds)-1)

        msk_inds_str=''.join([str(m) for m in msk_inds])

        tokens1=[[]]
        labels1=[]

        for j in range(len(words)):

            if j in msk_inds:

                wordtok=wordtoks[j]
                tokens1c=tokens1.copy()

                msk_inds_str1=msk_inds_str+'_'+str(j)

                tokens1=[tokens + [tokenizer.mask_token_id]*len(wordtok) for tokens in tokens1]
                
                tok_orders=[list(itertools.combinations(np.arange(len(wordtok)),x)) for x in range(1,len(wordtok))]             
                tok_orders = [list(item) for sublist in tok_orders for item in sublist]
                
                tokens2=[]
                
                for tok_order in tok_orders:                                       
                    for tokens in tokens1c[:1]:
                        for toki,tok in enumerate(wordtok):

                            if toki in tok_order:
                                tokens=tokens+[tok]
                            else:
                                tokens=tokens+[tokenizer.mask_token_id]

                        tokens2.append(tokens)
                            
                tokens1=tokens1+tokens2
            
                
                if len(wordtok)>1:

                    perms=list(itertools.permutations(np.arange(len(wordtok)),len(wordtok)))

                    input_to_mask_inds[msk_inds_str1]=[]

                    for perm in perms:
                        
                        temprows=[]

                        perm=list(perm)

                        for pi in range(len(perm)):

                            perm1=perm[:pi]
                            perm1sort=list(np.sort(perm1))
                            
                            if len(perm1sort)==0:
                                
                                row1=len(tokens_all)                             
                                row2=len(tokens1c[0])+perm[pi]
                                
                            else:

                                row1_offset=tok_orders.index(perm1sort)+1                                  
                                row1=len(tokens_all)+row1_offset
                                row2=len(tokens1c[0])+perm[pi]
                                
                                
                            row3=row2                            
                            rows=[row1,row2,row3]                        
                            temprows.append(rows)
                            
                        input_to_mask_inds[msk_inds_str1].append(temprows)

                            
                else:
                    
                    row1=len(tokens_all)
                    row2=len(tokens1c[0])
                    row3=row2
                
                    rows=[row1,row2,row3]
                    
                    input_to_mask_inds[msk_inds_str1]=[[rows]]                
                        
            else:

                tokens1=[tokens+wordtoks[j] for tokens in tokens1]

                
                
        tokens_all=tokens_all+tokens1



    tokens_all=[[tokenizer.cls_token_id]+t+[tokenizer.encode('.')[1:-1][0],tokenizer.sep_token_id] for t in tokens_all]
    
    inputs=torch.tensor(tokens_all).to('cuda')#.unsqueeze(0)

    batchsize=1000
    
    with torch.no_grad():

        if len(inputs)<batchsize:
  
            out = model(input_ids=inputs)[0]#.cpu().data.numpy()
    
            out=out[:,1:-2,:]
    
            for x in range(out.shape[1]):
                if x in start_inds[1:]:
                    out[:,x-1,suffs]=math.inf*-1
                elif x in suff_inds[1:]:
                    out[:,x-1,starts]=math.inf*-1
                    
            soft=torch.softmax(out,-1)
            
            soft=soft[:,:,word_tokens]

        else:  

            for b in range(int(np.ceil(len(inputs)/batchsize))):
                in1=inputs[batchsize*b:batchsize*(b+1)]
                lab1=labels_all[batchsize*b:batchsize*(b+1)]
                out1 = model(input_ids=in1)[0]#, masked_lm_labels=lab1)#.cpu().data.numpy()
                
                out1=out1[:,1:-2,:]
                
                for x in range(out1.shape[1]):
                    if x in start_inds[1:]:
                        out1[:,x-1,suffs]=math.inf*-1
                    elif x in suff_inds[1:]:
                        out1[:,x-1,starts]=math.inf*-1

                soft1=torch.softmax(out1,-1)
                
                soft1=soft1[:,:,word_tokens]
                
          
                if b==0:
                    soft=soft1
                    
                else:
                    soft=torch.cat((soft,soft1))
                    
                torch.cuda.empty_cache()


    

        orders=list(itertools.permutations(word_inds,i))

        orders=random.sample(orders,100)

        for orderi,order in enumerate(orders):

            for ordi,ind in enumerate(order):

                curr_masked=np.sort(order[ordi:])

                key=''.join([str(c-1) for c in curr_masked])+'_'+str(ind-1) #-1 to correct for CLS

                out_inds_all=input_to_mask_inds[key]

                for oi_all,out_inds in enumerate(out_inds_all):

                    for oi,out_ind in enumerate(out_inds):

                        prob=soft[out_ind[0],out_ind[1],out_ind[2]]

                        if oi==0:
                            word_probs=prob.unsqueeze(0)
                        else:
                            word_probs = torch.cat((word_probs, prob.unsqueeze(0)), 0)


                    word_probs_prod=torch.prod(word_probs)

                    if oi_all==0:
                        word_probs_all=word_probs_prod.unsqueeze(0)
                    else:
                        word_probs_all = torch.cat((word_probs_all, word_probs_prod.unsqueeze(0)), 0)


                word_prob=torch.mean(word_probs_all)

                if ordi==0:
                    chain_prob=word_prob.unsqueeze(0)
                else:
                    chain_prob = torch.cat((chain_prob, word_prob.unsqueeze(0)), 0)

            chain_prob_prod=torch.sum(torch.log(chain_prob))

            if chain_prob_prod==0:
                sys.exit()

            if orderi==0:
                chain_probs=chain_prob_prod.unsqueeze(0)
            else:
                chain_probs = torch.cat((chain_probs, chain_prob_prod.unsqueeze(0)), 0)

        
        score=float(np.exp(float(np.mean(chain_probs.cpu().data.numpy())))*float(per_cent))
        
        
        return score
