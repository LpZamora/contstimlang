import pickle
import numpy as np
import random
import math
import argparse

from tqdm import tqdm

from lstm_class import RNNLM
from bilstm_class import RNNLM_bilstm
from rnn_class import RNNModel
from model_functions import model_factory

parser = argparse.ArgumentParser()
parser.add_argument("--model",required=True)
args = parser.parse_args()

model=args.model

model1=model_factory(model,0)

file=open('sents_reddit_natural_May2021_filtered.txt','r')
sents=file.read()
sents=sents.split('\n')

probs=[]
for sent in tqdm(sents):
    prob=model1.sent_prob(sent)
    probs.append(prob)

probs=np.array(probs)

np.save('sents_reddit_natural_May2021_filtered_probs_'+model+'.npy',probs)