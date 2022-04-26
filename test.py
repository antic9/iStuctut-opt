import argparse
import os
import shutil
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from vocab import Vocab
from model import *
from utils import *
from batchify import get_batches
from train import evaluate

vocab = ''
device = ''
model = ''


def get_model(path):
    ckpt = torch.load(path)
    train_args = ckpt['args']
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[train_args.model_type](
        vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.flatten()
    model.eval()
    return model

def decode(z):
    sents = []
    i = 0
    while i < len(z):
        zi = torch.tensor(z[i: i+256], device=device)
        outputs = model.generate(zi, 35, 'greedy').t()
        for s in outputs:
            sents.append([vocab.idx2word[id] for id in s[1:]])  # skip <go>
        i += 256
    return strip_eos(sents)

def decodez(data):
    global vocab,device,model
    # print(data)
    vocab = Vocab(os.path.join('../checkpoints/CNN4MNIST/dimension10_temp_ns', 'vocab.txt'))
    set_seed(1111)
    cuda = not True and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = get_model(os.path.join('../checkpoints/CNN4MNIST/dimension10_temp_ns', 'model.pt'))
    z = data
    sents_rec = decode(z)
    return sents_rec
    