# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = "satoshi tsutsui"

import numpy as np
import tensorflow as tf
import ad_hoc_functions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--graph', type=str, default="../data/co-author-matrix.npz",help=u"numpy serialized scipy.sparse.csr_matrix. See http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format")
parser.add_argument('--walks', type=str, default='../work/random_walks.npz' ,help=u"numpy serialized random walks. Use codes/pre_cumpute_walks.py to generate this file")
parser.add_argument('--log', type=str, default="../log1/", help=u"directory to save tensorflow logs")
parser.add_argument('--save', type=str, default='../results/node_embeddings.npz', help=u"directory to save final embeddigs")
args = parser.parse_args()

print("loading adjacent matrix")

file_csr_matrix=args.graph
adj_mat_csr_sparse=ad_hoc_functions.load_sparse_csr(file_csr_matrix)

print("loading pre-computed random walks")
random_walk_files=args.walks
np_random_walks=np.load(random_walk_files)['arr_0']
np.random.shuffle(np_random_walks)

print("defining compuotational graphs")
#Computational Graph Definition
num_nodes=adj_mat_csr_sparse.shape[0]
context_size=16
batch_size = None
embedding_size = 200 # Dimension of the embedding vector.
num_sampled = 64 # Number of negative examples to sample.

global_step = tf.Variable(0, name='global_step', trainable=False)

# Parameters to learn
node_embeddings = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0))

#Fixedones
biases=tf.zeros([num_nodes])

# Input data and re-orgenize size.
with tf.name_scope("context_node") as scope:
    #context nodes to each input node in the batch (e.g [[1,2],[4,6],[5,7]] where batch_size = 3,context_size=3)
    train_context_node= tf.placeholder(tf.int32, shape=[batch_size,context_size],name="context_node")
    #orgenize prediction labels (skip-gram model predicts context nodes (i.e labels) given a input node)
    #i.e make [[1,2,4,6,5,7]] given context above. The redundant dimention is just for restriction on tensorflow API.
    train_context_node_flat=tf.reshape(train_context_node,[-1,1])
with tf.name_scope("input_node") as scope:
    #batch input node to the network(e.g [2,1,3] where batch_size = 3)
    train_input_node= tf.placeholder(tf.int32, shape=[batch_size],name="input_node")
    #orgenize input as flat. i.e we want to make [2,2,2,1,1,1,3,3,3] given the  input nodes above
    input_ones=tf.ones_like(train_context_node)
    train_input_node_flat=tf.reshape(tf.mul(input_ones,tf.reshape(train_input_node,[-1,1])),[-1])

# Model.
with tf.name_scope("loss") as scope:
    # Look up embeddings for words.
    node_embed = tf.nn.embedding_lookup(node_embeddings, train_input_node_flat)
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss_node2vec = tf.reduce_mean(tf.nn.sampled_softmax_loss(node_embeddings,biases,node_embed,train_context_node_flat, num_sampled, num_nodes))
    loss_node2vec_summary = tf.scalar_summary("loss_node2vec", loss_node2vec)

# Initializing the variables
init = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver(max_to_keep=20)

# Optimizer.
update_loss = tf.train.AdamOptimizer().minimize(loss_node2vec,global_step=global_step)

merged = tf.merge_all_summaries()

num_random_walks=np_random_walks.shape[0]
random_walk_length=np_random_walks.shape[1]

# Launch the graph
# Initializing the variables
init = tf.initialize_all_variables()

print("Optimizing")
with tf.Session() as sess:
    log_dir=args.log# tensorboard --logdir=./log1
    writer = tf.train.SummaryWriter(log_dir, sess.graph)
    sess.run(init)
    for i in xrange(0,num_random_walks):
        a_random_walk=np_random_walks[i]
        train_input_batch = np.array([a_random_walk[j] for j in xrange(random_walk_length-context_size)])
        train_context_batch = np.array([a_random_walk[j+1:j+1+context_size] for j in xrange(random_walk_length-context_size)])
        feed_dict={train_input_node:train_input_batch,
                   train_context_node:train_context_batch,}        
        _,loss_value,summary_str=sess.run([update_loss,loss_node2vec,merged], feed_dict)
        writer.add_summary(summary_str,i)

        with open(log_dir+"loss_value.txt", "a") as f:
            f.write(str(loss_value)+'\n') 
                
        # Save the variables to disk.
        if i%10000==0:
            model_path=log_dir+"model.ckpt"
            save_path = saver.save(sess, model_path,global_step)
            print("Model saved in file: %s" % save_path)

    model_path=log_dir+"model.ckpt"
    save_path = saver.save(sess, model_path,global_step)
    print("Model saved in file: %s" % save_path)

    print("Save final embeddings as numpy array")
    np_node_embeddings=sess.run(node_embeddings)
    np.savez(args.save,np_node_embeddings)
