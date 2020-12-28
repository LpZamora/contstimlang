import torch

# can we move these out of main?
from lstm_class import RNNLM
from bilstm_class import RNNLM_bilstm
from rnn_class import RNNModel

from model_functions import model_factory
from sentence_optimization import optimize_sentence_set, human_choice_controversiality_loss, human_choice_probability

n_sentences=2 # we optimize a pair of sentences
sent_len=8 # 8 words per sentence

# model names
model1_name='bert_whole_word'
model2_name='gpt2'

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
human_choice_response_models=None
model_names=None

model_prior=None

def loss_func(sentences_log_p):
    return human_choice_controversiality_loss(sentences_log_p,human_choice_response_models=human_choice_response_models,model_names=model_names,model_prior=model_prior)

#internal_stopping_condition=lambda loss: False # don't stop optimizing until convergence
internal_stopping_condition=lambda loss: loss<-0.999 

external_stopping_check=lambda: False # TODO- replace with file check

# prepare external stopping conditions
results=optimize_sentence_set(n_sentences,models=[model1,model2],loss_func=loss_func,sent_len=sent_len,
                         initial_sampling='uniform',external_stopping_check=external_stopping_check,
                         internal_stopping_condition=internal_stopping_condition,
                         start_with_identical_sentences=True, max_steps=10000,
                         max_replacements_to_attempt_per_word=50,
                         verbose=3)


human_p=human_choice_probability(results['sentences_log_p'])

