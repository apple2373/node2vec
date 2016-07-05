# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = "satoshi tsutsui"

import numpy as np
import ad_hoc_functions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--graph', type=str, default="../data/co-author-matrix.npz",help=u"numpy serialized scipy.sparse.csr_matrix. See http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format")
parser.add_argument('--walks', type=str, default='../work/random_walks.npz' ,help=u"path to save numpy serialized random walks.")
parser.add_argument('--p', type=float, default=1.0 ,help=u"Node2vec parameter p")
parser.add_argument('--q', type=float, default=0.5 ,help=u"Node2vec parameter q")
args = parser.parse_args()

print("loading adjacent matrix")

file_csr_matrix=args.graph
adj_mat_csr_sparse=ad_hoc_functions.load_sparse_csr(file_csr_matrix)
p=args.p
q=args.q

print("computing transition probabilities")
transition = ad_hoc_functions.compute_transition_prob(adj_mat_csr_sparse,p,q)

print("generating random walks")
random_walk_length=100
random_walks = ad_hoc_functions.generate_random_walks(adj_mat_csr_sparse,transition,random_walk_length)

#This is only for one epoch. If you want to generate two epochs:
# random_walks1 = ad_hoc_functions.generate_random_walks(adj_mat_csr_sparse,transition,random_walk_length)
# random_walks2 = ad_hoc_functions.generate_random_walks(adj_mat_csr_sparse,transition,random_walk_length)
# random_walks=random_walks1.extend(random_walks2)

np_random_walks=np.array(random_walks,dtype=np.int32)
np.savez(args.graph.walks,np_random_walks)
