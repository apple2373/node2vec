# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import ad_hoc_functions


print("loading adjacent matrix")

file_csr_matrix="../data/co-author-matrix.npz"
adj_mat_csr_sparse=ad_hoc_functions.load_sparse_csr(file_csr_matrix)
p=1.0
q=0.5

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
np.savez('../work/random_walks.npz',np_random_walks)
