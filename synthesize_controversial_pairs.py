import os

import torch

# can we move these out of main?
from lstm_class import RNNLM
from bilstm_class import RNNLM_bilstm
from rnn_class import RNNModel

from model_functions import model_factory
from sentence_optimization import optimize_sentence_set, human_choice_controversiality_loss, human_choice_controversiality_loss_log_scale, human_choice_probability
from model_to_human_decision_torch import load_decision_model, Naive

n_sentences=2 # we optimize a pair of sentences
sent_len=8 # 8 words per sentence

# model names
model1_name='bigram'
model2_name='gpt2'

sentences=['Ira has afield interview withstanding for someplace toasted','There used profanity Pueblo midterms behind him that']

#bigram and trigram models run on CPU, so gpu_id will be ignored
if torch.cuda.device_count()>=2:
    model1_gpu_id=0
    model2_gpu_id=1
else:
    model1_gpu_id=0
    model2_gpu_id=0

# load models
model1=model_factory(model1_name,model1_gpu_id)
model2=model_factory(model2_name,model2_gpu_id)

# load response models
decision_model_class='SquashedSoftmax'
optimizer='LBFGS'
human_choice_response_models=[]
for model_name in [model1_name,model2_name]:
    path = os.path.join('decision_models','20201228',decision_model_class+"_" +optimizer,model_name+'.pkl')
    human_choice_response_models.append(load_decision_model(path,device='cpu'))    
    #human_choice_response_models.append(Naive())
model_prior=None


def loss_func(sentences_log_p):
    return human_choice_controversiality_loss_log_scale(sentences_log_p,human_choice_response_models=human_choice_response_models,model_prior=model_prior)

def monitoring_func(sentences,sentences_log_p):
    human_p=human_choice_probability(sentences_log_p,human_choice_response_models=human_choice_response_models,log_scale=False)    
    print(model1_name+":"+'{:.2f}/{:.2f}'.format(human_p[0,0,0],human_p[0,0,1]))
    print(model2_name+":"+'{:.2f}/{:.2f}'.format(human_p[0,1,0],human_p[0,1,1]))

internal_stopping_condition=lambda loss: False # don't stop optimizing until convergence

external_stopping_check=lambda: False # TODO- replace with file check

# prepare external stopping conditions
results=optimize_sentence_set(n_sentences,models=[model1,model2],loss_func=loss_func,sentences=sentences,sent_len=sent_len,
                         initial_sampling='uniform',external_stopping_check=external_stopping_check,
                         monitoring_func=monitoring_func,
                         internal_stopping_condition=internal_stopping_condition,
                         start_with_identical_sentences=True, max_steps=10000,
                         verbose=3)
print(sentences)
monitoring_func(results['sentences'],results['sentences_log_p'])


