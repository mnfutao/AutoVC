#!/bin/bash

python new_inference.py logdir/model-290000.pt data/new_char_emb.npy 'biaobei' 'yaya' results/biaobei-010000.npy  results

#python new_inference.py logdir/model-290000.pt data/new_char_emb.npy 'biaobei' 'snowball_v2' results/biaobei-010000.npy  results

#python new_inference.py logdir/model-210000.pt data/new_char_emb.npy 'biaobei' 'speechocean_man10h' results/biaobei-010000.npy  results
