# contstim
code for generating controversial stimuli

# environment setup

Python 3.7.6

CUDA Version: 10.2 

# how to install

-> git clone https://github.com/dpmlab/contstim.git

-> cd contstim

-> pip install -r requirements.txt

-> bash download_models.sh (This will download the following models from google drive: BIGRAM, TRIGRAM, RNN, LSTM, BILSTM. The transformer models will automatically download when the generation script is run.)

# how to generate a controversial synthetic sentence pair

Use the file **make_contstim** to generate controversial sentence pairs. There are two versions of this file – a python notebook and a python script. Both should work. There are three arguments that need to be hardcoded at the top of the scripts: model1, model2, and squashing_threshold.

# how to generate an entire set of synthetic controversial sentence pairs
Run synthesize_controversial_pairs_batch_job.py. This file can be run in parallel by multiple nodes/workers. Each sentence pair can take a few minutes, so multiple nodes would be needed for completing a large sentence pair set in a reasonable time.

# currently included models 

GPT2

BERT

ROBERTA

ELECTRA

XLM

BILSTM

LSTM

RNN

TRIGRAM

BIGRAM

BERT_WHOLE_WORD (not evaluated in the paper)