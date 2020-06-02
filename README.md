# contstim
code for generating controversial stimuli

# environment setup

Python 3.7.6

CUDA Version: 10.2 

# how to install

-> git clone https://github.com/dpmlab/contstim.git

-> cd contstim

-> pip install -r requirements.txt

-> bash download_models.sh (This will download the rnn and lstm models from google drive. The transformer models will automatically download when the generation script is run.)

# how to generate controversial stimuli

Use the file **make_contstim** to generate controversial sentence pairs. There are two versions of this file – a python notebook and a python script. Both should work. There are three arguments that need to be hardcoded at the top of the scripts: model1, model2, and squashing_threshold.

# models 

gpt2

bert

bert_whole_word

roberta

electra

xlm

lstm

rnn

coming soon: bigram & trigram
