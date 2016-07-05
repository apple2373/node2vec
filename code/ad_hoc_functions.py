# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = "satoshi tsutsui"

import numpy as np
from scipy.sparse import csr_matrix
import multiprocessing as mp
import json

#ref http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),shape = loader['shape'])

def alpha(p,q,t,x,adj_mat_csr_sparse):
    if t==x:
        return 1.0/p
    elif adj_mat_csr_sparse[t,x]>0:
        return 1.0
    else:
        return 1.0/q

def compute_transition_prob(adj_mat_csr_sparse,p,q):
    transition={}
    num_nodes=adj_mat_csr_sparse.shape[0]
    indices=adj_mat_csr_sparse.indices
    indptr=adj_mat_csr_sparse.indptr
    data=adj_mat_csr_sparse.data
    #Precompute the transition matrix in advance
    for t in xrange(num_nodes):#t is row index
        for v in indices[indptr[t]:indptr[t+1]]:#i.e  possible next ndoes from t
            pi_vx_indices=indices[indptr[v]:indptr[v+1]]#i.e  possible next ndoes from v
            pi_vx_values = np.array([alpha(p,q,t,x,adj_mat_csr_sparse) for x in pi_vx_indices])
            pi_vx_values=pi_vx_values*data[indptr[v]:indptr[v+1]]
            #This is eqilvalent to the following
    #         pi_vx_values=[]
    #         for x in pi_vx_indices:
    #             pi_vx=alpha(p,q,t,x)*adj_mat_csr_sparse[v,x]
    #             pi_vx_values.append(pi_vx)
            pi_vx_values=pi_vx_values/np.sum(pi_vx_values)
            #now, we have normalzied transion probabilities for v traversed from t
            #the probabilities are stored as a sparse vector. 
            transition[t,v]=(pi_vx_indices,pi_vx_values)

    return transition


def generate_random_walks(adj_mat_csr_sparse,transition,random_walk_length):
    random_walks=[]
    num_nodes=adj_mat_csr_sparse.shape[0]
    indices=adj_mat_csr_sparse.indices
    indptr=adj_mat_csr_sparse.indptr
    data=adj_mat_csr_sparse.data
    #get random walks
    for u in xrange(num_nodes):
        if len(indices[indptr[u]:indptr[u+1]]) !=0:
            #first move is just depends on weight
            possible_next_node=indices[indptr[u]:indptr[u+1]]
            weight_for_next_move=data[indptr[u]:indptr[u+1]]#i.e  possible next ndoes from u
            weight_for_next_move=weight_for_next_move.astype(np.float32)/np.sum(weight_for_next_move)
            first_walk=np.random.choice(possible_next_node, 1, p=weight_for_next_move)
            random_walk=[u,first_walk[0]]
            for i in xrange(random_walk_length-2):
                cur_node = random_walk[-1]
                precious_node=random_walk[-2]
                (pi_vx_indices,pi_vx_values)=transition[precious_node,cur_node]
                next_node=np.random.choice(pi_vx_indices, 1, p=pi_vx_values)
                random_walk.append(next_node[0])
            random_walks.append(random_walk)

    return random_walks
