# Node2vec with tensorflow
This repo contains ad hoc implementation of node2vec using tensorflow. I call it ad hoc because the codes are not so clean and efficient. However, it is applicable for large networks. I tested on a network with 74,530 nodes. Also, the input network needs to be represented as a Scipy.sparse.csr_matrix. 
  sa
main reference appeared at KDD 2016: [node2vec: Scalable Feature Learning for Networks](http://aditya-grover.github.io/files/publications/kdd16.pdf)

## Requirements
- [tneosorflow 0.9](http://tensorflow.org)

## How to use.
I constructed a co-author network from Microsoft academic graph
(https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/). This is co-author network only from SIGMOD, VLDB, ICDE, ICML, NIPS, IJCAI, AAAI, ECMLPAKDD, ICCV, CVPR, ECCV, ACL, NAACL, EMNLP, KDD, ICDM, WSDM, WWW, CIKM, and  ISWC. It has 74,530 nodes.  I’ll use this example here.

First, prepare sample nodes by random walk parameterized with p (return parameter) and q (in-out parameter). The input network has to be scipy.sparse.csr_matrix represented as selizalized as noted [here](http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format) 
```
cd code #make sure you are in code directory
python pre_cumpute_walks.py --graph ../data/co-author-matrix.npz --walk ../work/random_walks.npz --p 1.0 --q 0.5
```
Then, learn embeddings uisng the random walks
```
cd code #make sure you are in code directory
python train_node2vec.py --graph ../data/co-author-matrix.npz --walk ../work/random_walks.npz --log ./log1/ --save ../results/node_embeddings.npz
```

## Download  network, random walks, and embeddings
I made a sample random walks, and learned embeddings available. Let's downlaod it. 
```
python download_model.py
```
This will create these files : fast_rcnn_vgg_voc.model
If you want to download manually,
https://googledrive.com/host/0B046sNk0DhCDUk9YeklwOFczc0E/fast_rcnn_vgg_voc.model

## Vector Examples
See more examples on ipython notebook: 

##To do list
Make the code more flexible using command line arguments (e.g. dimensions of embeddings)
Use multi-processing for computing transition probabilities and random walks. 
Use asynchronous SGD (currently using Adam SGD with single process)