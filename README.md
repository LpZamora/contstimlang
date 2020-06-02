# contstim
code for generating controversial stimuli

# environment setup

Python 3.7.6

CUDA Version: 10.2 

# how to install

-> git clone https://github.com/dpmlab/contstim.git

-> cd contstim

-> pip install -r requirements.txt

-> bash download_models.sh

# how to generate controversial stimuli

Use the file **make_contstim** to generate controversial sentence pairs. There are two versions of this file – a python notebook and a python script. Both should work. There are three arguments that need to be hardcoded at the top of the scripts: i) model1, model2, and squashing_threshold.
