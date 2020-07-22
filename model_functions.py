from transformers import BertForMaskedLM, BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel, RobertaForMaskedLM, RobertaTokenizer, \
XLMTokenizer, XLMWithLMHeadModel, ElectraTokenizer, ElectraForMaskedLM
import numpy as np
import torch
import math
import itertools
import random
import pickle
from knlm import KneserNey

###############################################################

with open('vocab_low.pkl', 'rb') as file:
    vocab_low=pickle.load(file) 
    
with open('vocab_low_freqs.pkl', 'rb') as file:
    vocab_low_freqs=pickle.load(file) 

with open('vocab_cap.pkl', 'rb') as file:
    vocab_cap=pickle.load(file) 
    
with open('vocab_cap_freqs.pkl', 'rb') as file:
    vocab_cap_freqs=pickle.load(file) 

###############################################################
   
with open('neuralnet_word2id_dict.pkl', 'rb') as file:
    word2id=pickle.load(file)

nn_vocab_size=np.max([word2id[w] for w in word2id])+1

word2id['[MASK]'] = nn_vocab_size
id2word=dict(zip([word2id[w] for w in word2id],[w for w in word2id]))

######################################################## 


class model_factory:

    def __init__(self, name, gpu_id):
        
        self.name=name
        self.gpu_id=gpu_id
        
        if name=='bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
            self.model = BertForMaskedLM.from_pretrained('bert-large-cased').to('cuda:'+str(gpu_id))
            
        elif name=='bert_whole_word':
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
            self.model = BertForMaskedLM.from_pretrained('bert-large-cased-whole-word-masking').to('cuda:'+str(gpu_id))
            
        elif name=='roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            self.model = RobertaForMaskedLM.from_pretrained('roberta-large').to('cuda:'+str(gpu_id))
            
        elif name=='xlm':
            self.tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
            self.model = XLMWithLMHeadModel.from_pretrained('xlm-mlm-en-2048').to('cuda:'+str(gpu_id))
            
        elif name=='electra':
            self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-generator')
            self.model = ElectraForMaskedLM.from_pretrained('google/electra-large-generator').to('cuda:'+str(gpu_id))
            
        elif name=='gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to('cuda:'+str(gpu_id))
            
        elif name=='bilstm':       
            self.model=torch.load('contstim_bilstm.pt').to('cuda:'+str(gpu_id))
            self.word2id=word2id
            self.id2word=id2word
            self.embed_size=256
            self.hidden_size=256
            self.vocab_size=nn_vocab_size
            self.num_layers=1
            
        elif name=='lstm':     
            self.model=torch.load('contstim_lstm.pt').to('cuda:'+str(gpu_id))
            self.word2id=word2id
            self.id2word=id2word
            self.embed_size=256
            self.hidden_size=512
            self.vocab_size=nn_vocab_size
            self.num_layers=1
            
        elif name=='rnn':           
            self.model=torch.load('contstim_rnn.pt').to('cuda:'+str(gpu_id))
            self.word2id=word2id
            self.id2word=id2word
            self.embed_size=256
            self.hidden_size=512
            self.vocab_size=nn_vocab_size
            self.num_layers=1
            
            
        elif name=='trigram':           
            self.model=KneserNey.load('trigram.model')
            
        elif name=='bigram':           
            self.model=KneserNey.load('bigram.model')

            
                      
        self = get_starts_suffs(self)
        self = get_token_info(self)
        
        

    def sent_prob(self,sent):
        
        if self.name in ['bert','bert_whole_word','roberta','xlm','electra']:
            prob=bidirectional_transformer_sent_prob(self,sent)
          
        elif self.name == 'gpt2':
            prob=gpt2_sent_prob(self,sent)
            
        elif self.name == 'bilstm':
            prob=bilstm_sent_prob(self,sent)
            
        elif self.name == 'lstm':
            prob=lstm_sent_prob(self,sent)
            
        elif self.name == 'rnn':
            prob=rnn_sent_prob(self,sent)
            
        elif self.name == 'trigram':
            prob=trigram_sent_prob(self,sent)
        
        elif self.name == 'bigram':
            prob=bigram_sent_prob(self,sent)    
        
        return prob    

    def word_probs(self,words,wordi):

        if self.name in ['bert','bert_whole_word','roberta','xlm','electra']:
            probs=bidirectional_transformer_word_probs(self, words, wordi)
            
        elif self.name == 'gpt2':
            probs=gpt2_word_probs(self, words, wordi)
            
        elif self.name == 'bilstm':
            probs=bilstm_word_probs(self, words, wordi)
            
        elif self.name == 'lstm':
            probs=lstm_word_probs(self, words, wordi)
            
        elif self.name == 'rnn':
            probs=rnn_word_probs(self, words, wordi)
            
        elif self.name == 'trigram':
            probs=trigram_word_probs(self, words, wordi)
        
        elif self.name == 'bigram':
            probs=bigram_word_probs(self, words, wordi)  

        return probs

        
        
        
def get_starts_suffs(self):
    
    name=self.name
    
    if self.name in ['bilstm','lstm','rnn','bigram','trigram']:
        return self

    starts=[]
    suffs=[]
    
    tokenizer=self.tokenizer
    
    if name in ['bert','bert_whole_word','electra']:
        for i in range(len(tokenizer.get_vocab())):
            tok=tokenizer.decode(i)
            if tok[0]!='#':
                starts.append(i)
            elif tok[0]!=' ':
                suffs.append(i)
                
    elif name in ['gpt2','roberta']:      
        for i in range(len(tokenizer.get_vocab())):
            tok=tokenizer.decode(i)
            if tok[0]==' ' or tok[0]=='.':
                starts.append(i)
            elif tok[0]!=' ':
                suffs.append(i)
                
    elif name in ['xlm']:        
        for i in range(len(tokenizer.get_vocab())):
            tok=tokenizer.convert_ids_to_tokens(i)
            if tok[-4:]=='</w>' and tok!='.</w>':
                suffs.append(i)
            else:
                starts.append(i)

    self.starts=starts
    self.suffs=suffs
    
    return self     


    

def get_token_info(self):
    
    name=self.name
    
    if self.name in ['bilstm','lstm','rnn','bigram','trigram']:
        return self
    
    
    tokenizer=self.tokenizer
    model=self.model
    
    
    if name=='gpt2':
        
        special_tokens_dict = {'pad_token': '[PAD]'}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))  
        
        toklist_low=[]
        toklist_cap=[]
        
        for v in vocab_low:
            toks=tokenizer.encode(' '+v)
            toklist_low.append(toks)

        for v in vocab_cap:
            toks=tokenizer.encode(' '+v)
            toklist_cap.append(toks) 
            
        self.tokenizer=tokenizer
        self.model=model
        self.toklist_low=toklist_low
        self.toklist_cap=toklist_cap
           
    else:

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

            
            self.vocab_low=vocab_low
            self.unique_tokparts_low=unique_tokparts_low
            self.vocab_probs_sheet_low=vocab_probs_sheet_low
            self.vocab_to_tokparts_inds_map_low=vocab_to_tokparts_inds_map_low
            
            self.vocab_cap=vocab_cap
            self.unique_tokparts_cap=unique_tokparts_cap
            self.vocab_probs_sheet_cap=vocab_probs_sheet_cap
            self.vocab_to_tokparts_inds_map_cap=vocab_to_tokparts_inds_map_cap

            
        return self
    

def bidirectional_transformer_sent_prob(self,sent):
    
    gpu_id=self.gpu_id
    
    tokenizer=self.tokenizer
    model=self.model
    
    starts=self.starts
    suffs=self.suffs
 
    word_tokens_per=tokenizer.encode(sent+'.')
    word_tokens_per[-2]=tokenizer.mask_token_id
    in1=torch.tensor(word_tokens_per).to('cuda:'+str(gpu_id)).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=in1)[0]
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
    
    inputs=torch.tensor(tokens_all).to('cuda:'+str(gpu_id))#.unsqueeze(0)

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
    

    
def bidirectional_transformer_word_probs(self,words,wordi):
    
    gpu_id=self.gpu_id
    
    tokenizer=self.tokenizer
    model=self.model
    
    if wordi>0:
        vocab=self.vocab_low
        unique_tokparts=self.unique_tokparts_low
        vocab_probs_sheet=self.vocab_probs_sheet_low
        vocab_to_tokparts_inds_map=self.vocab_to_tokparts_inds_map_low
    else:
        vocab=self.vocab_cap
        unique_tokparts=self.unique_tokparts_cap
        vocab_probs_sheet=self.vocab_probs_sheet_cap
        vocab_to_tokparts_inds_map=self.vocab_to_tokparts_inds_map_cap
    
    
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

    inputs=torch.tensor(inputs).to('cuda:'+str(gpu_id))
    att_mask=torch.tensor(att_mask).to('cuda:'+str(gpu_id))

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


def gpt2_sent_prob(self,sent):
    
    gpu_id=self.gpu_id
    
    tokenizer=self.tokenizer
    model=self.model
    
    starts=self.starts
    suffs=self.suffs
    
    sent='. ' + sent + '.'
    
    tokens=tokenizer.encode(sent)
    inputs=torch.tensor(tokens).to('cuda:'+str(gpu_id))
    
    with torch.no_grad():
        out=model(inputs)

    unsoft=out[0]
    lab1=inputs.cpu().data.numpy()
    
    probs=[]
    for x in range(len(lab1)-1):
        
        lab=lab1[x+1]       
        unsoft1=unsoft[x]
        
        if lab in starts:
            
            soft=torch.softmax(unsoft1[starts],-1)            
            prob=float(soft[starts.index(lab)].cpu().data.numpy())
                      
        elif lab in suffs:
            
            soft=torch.softmax(unsoft1[suffs],-1)          
            prob=float(soft[suffs.index(lab)].cpu().data.numpy())
            
        probs.append(prob)
        
    prob=np.prod(probs)
    
    return prob

def gpt2_word_probs(self,words,wordi):
    
    gpu_id=self.gpu_id
    
    tokenizer=self.tokenizer
    model=self.model
    
    starts=self.starts
    suffs=self.suffs
    
    if wordi==0:
        vocab=vocab_cap
        toklist=self.toklist_cap
    else:
        vocab=vocab_low
        toklist=self.toklist_low

    
    sent1=' '.join(words[:wordi])
    sent2=' '.join(words[wordi+1:])

    tok1=tokenizer.encode('. ' + sent1)
    tok2=tokenizer.encode(' ' + sent2)

    ####################################################3## 
        
    lp=0
    while 0==0:
        in1=tok1
        in1=torch.tensor(in1).to('cuda:'+str(gpu_id))

        with torch.no_grad():
            out1=model(in1)[0]
            soft1=torch.softmax(out1,-1)[-1].cpu().data.numpy()

        logsoft1=np.log(soft1)

        tops=np.where(logsoft1>-10-lp*5)[0]

        tops=[t for t in tops if t in starts]
        
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

        inputs2=torch.tensor(inputs1).to('cuda:'+str(gpu_id))
        att_mask1=torch.tensor(att_mask1).to('cuda:'+str(gpu_id))
       
        
        with torch.no_grad():
            
            out1=model(input_ids=inputs2,attention_mask=att_mask1)[0]           

            #out_suff_inds=torch.where(torch.tensor(np.in1d(inputs1,suffs).reshape(inputs1.shape[0],-1)).to('cuda:'+str(gpu_id))==True)         
            #out_start_inds=torch.where(torch.tensor(np.in1d(inputs1,starts).reshape(inputs1.shape[0],-1)).to('cuda:'+str(gpu_id))==True) 

#             for x in range(len(out_suff_inds[0])):
#                 out1[out_suff_inds[0][x],out_suff_inds[1][x]-1,starts_cuda]=math.inf*-1

#             for x in range(len(out_start_inds[0])):
#                 out1[out_start_inds[0][x],out_start_inds[1][x]-1,suffs_cuda]=math.inf*-1

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





def bilstm_word_probs(self,words,wordi):
    
    gpu_id=self.gpu_id
    
    model=self.model
    
    hidden_size=self.hidden_size
    embed_size=self.embed_size
    vocab_size=self.vocab_size
    num_layers=self.num_layers
    
    states = (torch.zeros(2, 1, hidden_size).to('cuda:'+str(gpu_id)),
                  torch.zeros(2, 1, hidden_size).to('cuda:'+str(gpu_id)))

    ids=[word2id[w] for w in words]+[word2id['.']]

    ids[wordi]=word2id['[MASK]']

    ids=torch.tensor(ids).to('cuda:'+str(gpu_id)).unsqueeze(0)

    out, states = model(ids, states, 0, [wordi])

    soft=torch.softmax(out[0],-1).cpu().data.numpy()
    
    return soft



def bilstm_sent_prob(self,sent):
    
    gpu_id=self.gpu_id
    
    model=self.model
    
    hidden_size=self.hidden_size
    embed_size=self.embed_size
    vocab_size=self.vocab_size
    num_layers=self.num_layers
    
    words=sent.split() 

    word_ids=[word2id[w] for w in words]

    tok_orders=[list(itertools.combinations(np.arange(len(words)),x)) for x in range(1,len(words))]  
    tok_orders = ['']+[item for sublist in tok_orders for item in sublist]

    chains=[]
    order_to_chain=dict()

    for i, tok_order in enumerate(tok_orders):

        base=[word2id['[MASK]']]*len(words) + [word2id['.']]

        for t in tok_order:
            base[t]=word_ids[t]

        chains.append(base)

        key=''.join([str(t) for t in tok_order])

        order_to_chain[key]=i
        
        
    chains=torch.tensor(chains).to('cuda:'+str(gpu_id))
   
    states = (torch.zeros(2, chains.shape[0], hidden_size).to('cuda:'+str(gpu_id)),
                      torch.zeros(2, chains.shape[0], hidden_size).to('cuda:'+str(gpu_id)))

    
    out, states = model(chains, states, 0, np.arange(chains.shape[0]*chains.shape[1]))

    soft=torch.softmax(out,-1)

    soft=soft[:,word_ids]

    soft=soft.reshape(chains.shape[0],chains.shape[1],soft.shape[1])

    
    tok_perms=list(itertools.permutations(np.arange(len(words))))

    tok_perms100=random.sample(tok_perms,100)

    probs_all=[]

    for tok_perm in tok_perms100:

        probs=[]

        for tpi,tp in enumerate(tok_perm):

            key=''.join([str(t) for t in np.sort(tok_perm[:tpi])])

            key_ind=order_to_chain[key]

            chain=chains[key_ind]

            prob=float(torch.prod(soft[key_ind,tp,tp]))

            probs.append(prob)

        probs_all.append(np.prod(probs))
        
    prob=np.mean(probs_all)
    
    return prob
    
    
    
    
def lstm_word_probs(self,words,wordi):
    
    gpu_id=self.gpu_id
    
    model=self.model
    
    hidden_size=self.hidden_size
    embed_size=self.embed_size
    vocab_size=self.vocab_size
    num_layers=self.num_layers

    if wordi>0:
        vocab=vocab_low
    else:
        vocab=vocab_cap

    wordi=wordi+1

    words=['.'] + words + ['.']

    states = (torch.zeros(num_layers, 1, hidden_size).to('cuda:'+str(gpu_id)),
                          torch.zeros(num_layers, 1, hidden_size).to('cuda:'+str(gpu_id)))

    inputs = torch.tensor([word2id[w] for w in words]).to('cuda:'+str(gpu_id)).unsqueeze(0)
    outputs, states = model(inputs, states, 0)
    soft=torch.softmax(outputs,-1).cpu().data.numpy()

    ss = np.argsort(soft[wordi-1])[::-1]
    top_words=[id2word[s] for s in ss[:3000]]
    top_words=list(set(top_words)&set(vocab))
    inds=[vocab.index(t) for t in top_words]


    probs=[]

    for wi,w in enumerate(top_words):

        states = (torch.zeros(num_layers, 1, hidden_size).to('cuda:'+str(gpu_id)),
                          torch.zeros(num_layers, 1, hidden_size).to('cuda:'+str(gpu_id)))

        words[wordi]=w

        inputs = torch.tensor([word2id[w] for w in words]).to('cuda:'+str(gpu_id)).unsqueeze(0)

        outputs, states = model(inputs, states, 0)


        soft=torch.softmax(outputs,-1).cpu().data.numpy()

        prob=float(np.prod([float(soft[wordi-1+x,word2id[w1]]) for x,w1 in enumerate(words[wordi:])]))

        probs.append(prob)

    probs=np.array(probs)

    return probs, inds
  
    
    
    
    
    
def lstm_sent_prob(self,sent): 
    
    gpu_id=self.gpu_id
    
    model=self.model
    
    hidden_size=self.hidden_size
    embed_size=self.embed_size
    vocab_size=self.vocab_size
    num_layers=self.num_layers
    
    states = (torch.zeros(num_layers, 1, hidden_size).to('cuda:'+str(gpu_id)),
                  torch.zeros(num_layers, 1, hidden_size).to('cuda:'+str(gpu_id)))

    words=['.'] + sent.split() + ['.']

    inputs = torch.tensor([word2id[w] for w in words]).to('cuda:'+str(gpu_id)).unsqueeze(0)

    outputs, states = model(inputs, states, 0)

    soft=torch.softmax(outputs,-1).cpu().data.numpy()

    prob=float(np.prod([float(soft[wi,word2id[w]]) for wi,w in enumerate(words)]))

    return prob





def rnn_sent_prob(self,sent):   
    
    gpu_id=self.gpu_id
    
    model=self.model
    
    hidden_size=self.hidden_size
    embed_size=self.embed_size
    vocab_size=self.vocab_size
    num_layers=self.num_layers

    states = (torch.zeros(num_layers, 1, hidden_size).to('cuda:'+str(gpu_id)),
                  torch.zeros(num_layers, 1, hidden_size).to('cuda:'+str(gpu_id)))

    words=['.'] + sent.split() + ['.']

    inputs = torch.tensor([word2id[w] for w in words]).to('cuda:'+str(gpu_id)).unsqueeze(0)
    
    h0 = torch.zeros(num_layers, 1, hidden_size).to('cuda:'+str(gpu_id))

    outputs, states = model(inputs, h0)

    soft=torch.softmax(outputs,-1).cpu().data.numpy()[0]

    prob=float(np.prod([float(soft[wi,word2id[w]]) for wi,w in enumerate(words[1:])]))

    return prob


def rnn_word_probs(self,words,wordi):
    
    gpu_id=self.gpu_id
    
    model=self.model
    
    hidden_size=self.hidden_size
    embed_size=self.embed_size
    vocab_size=self.vocab_size
    num_layers=self.num_layers

    if wordi>0:
        vocab=vocab_low
    else:
        vocab=vocab_cap


    wordi=wordi+1

    words=['.'] + words + ['.']

    h0 = torch.zeros(num_layers, 1, hidden_size).to('cuda:'+str(gpu_id))


    inputs = torch.tensor([word2id[w] for w in words]).to('cuda:'+str(gpu_id)).unsqueeze(0)
    outputs, states = model(inputs, h0)
    soft=torch.softmax(outputs,-1).cpu().data.numpy()[0]

    ss = np.argsort(soft[wordi-1])[::-1]
    top_words=[id2word[s] for s in ss[:3000]]
    top_words=list(set(top_words)&set(vocab))
    inds=[vocab.index(t) for t in top_words]


    probs=[]

    for wi,w in enumerate(top_words):

        states = (torch.zeros(num_layers, 1, hidden_size).to('cuda:'+str(gpu_id)),
                          torch.zeros(num_layers, 1, hidden_size).to('cuda:'+str(gpu_id)))

        words[wordi]=w

        inputs = torch.tensor([word2id[w] for w in words]).to('cuda:'+str(gpu_id)).unsqueeze(0)

        outputs, states = model(inputs, h0)


        soft=torch.softmax(outputs,-1).cpu().data.numpy()[0]

        prob=float(np.prod([float(soft[wordi-1+x,word2id[w1]]) for x,w1 in enumerate(words[wordi:])]))

        probs.append(prob)

    probs=np.array(probs)

    return probs, inds

def trigram_sent_prob(self,sent):
    
    words=sent.split()
    
    model=self.model
    
    words=['<BOS1>','<BOS2>']+words+['.','<EOS1>']
    
    prob=np.exp(model.evaluateSent(words))
   
    return prob

def trigram_word_probs(self,words,wordi):
    
    model=self.model
    
    words=['<BOS1>','<BOS2>']+words+['.','<EOS1>']
    
    if wordi==0:
        vocab=vocab_cap
    else:
        vocab=vocab_low
        
    probs=[]
    for w in vocab:
        
        words[wordi+2]=w
        
        prob=np.exp(model.evaluateSent(words))
        
        probs.append(prob)
        
    probs=np.array(probs)
                   
    return probs
        
def bigram_sent_prob(self,sent):
    
    words=sent.split()
    
    model=self.model
    
    words=['<BOS2>']+words+['.']
    
    prob=np.exp(model.evaluateSent(words))
   
    return prob

def bigram_word_probs(self,words,wordi):
    
    model=self.model
    
    words=['<BOS2>']+words+['.']
    
    if wordi==0:
        vocab=vocab_cap
    else:
        vocab=vocab_low
        
    probs=[]
    for w in vocab:
        
        words[wordi+1]=w
        
        prob=np.exp(model.evaluateSent(words))
        
        probs.append(prob)
        
    probs=np.array(probs)
                   
    return probs       
        
        