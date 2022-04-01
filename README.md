# contstimlang
Code for generating controversial sentence pairs and supporting material for "Testing the limits of natural language models for predicting human language judgments"

# environment setup

Python 3.7.6

CUDA Version: 10.2 

# how to install (Anaconda, recommended)

-> git clone https://github.com/dpmlab/contstim.git

-> cd contstim

-> conda env create --name contstimlang --file contstimlang

-> python download_checkpoints.py 
(This will download the checkpoints for the following models from Zenodo: BIGRAM, TRIGRAM, RNN, LSTM, BILSTM. The transformer models will automatically download when the sentence generation code is first run.)

if you don't use Anaconda, replace `conda env create --name contstimlang --file contstimlang` with 

-> pip install requirements.txt

# how to generate a single controversial synthetic sentence pair
Use the file `make_contstim` to generate controversial sentence pairs. There are two versions of this file – a python notebook and a python script. Both should work. There are three arguments that need to be hardcoded at the top of the scripts: model1, model2, and squashing_threshold.

# how to reproduce the paper's figures
Run the file `behav_exp_analysis.py'.

# how to generate an entire set of synthetic controversial sentence pairs
Run `synthesize_controversial_pairs_batch_job.py`. This file is designed to be run in parallel by multiple nodes/workers.
We include the code we used as a reference, but it is not practical to use without an HPC environment, as each sentence pair can take a few minutes.
Each compute node should have 2 GPUs.

# how to generate an entire set of natural controversial sentence pairs
First, install GUROBI (https://www.gurobi.com/downloads/, there's a free academic license).
Then, run ``
The code takes about an hour on a modern workstation and may require high RAM (tested on a 128GB machine).

# currently included models 
GPT2, BERT, ROBERTA, ELECTRA, XLM, BILSTM, LSTM, RNN, TRIGRAM, BIGRAM, BERT_WHOLE_WORD (not evaluated in the paper)

# How to cite:
