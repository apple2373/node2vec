# Node2vec with tensorflow
This repo contains ad hoc implementation of node2vec using tensorflow. I call it ad hoc because the codes are not so clean and efficient. However, it is applicable for large networks. I tested on a network with 74,530 nodes. Also, the input network needs to be represented as a Scipy.sparse.csr_matrix. 
  
main reference appeared at KDD 2016: [node2vec: Scalable Feature Learning for Networks](http://arxiv.org/abs/1607.00653)
  
Also, I noticed that the first author of the paper open sourced the implementation: https://github.com/aditya-grover/node2vec
I guess that is more efficent. So please try to use that first. This repo is for people who want to use tensorflow for some reasons. 
  
## Requirements
I recommend you to install [Anaconda](https://www.continuum.io/downloads) and then tensorflow.
- [tneosorflow 0.9](http://tensorflow.org)
- and some other libraries...

## How to use.
I constructed a co-author network from [Microsoft academic graph](https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/). This is co-author network only from SIGMOD, VLDB, ICDE, ICML, NIPS, IJCAI, AAAI, ECMLPAKDD, ICCV, CVPR, ECCV, ACL, NAACL, EMNLP, KDD, ICDM, WSDM, WWW, CIKM, and  ISWC. It has 74,530 nodes.  I’ll use this example here. The data is in the ./data directory.

First, prepare sample nodes by random walk parameterized with p (return parameter) and q (in-out parameter). The input network has to be scipy.sparse.csr_matrix represented as serialized as noted [here](http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format) 
```
cd code #make sure you are in code directory
python pre_compute_walks.py --graph ../data/co-author-matrix.npz --walk ../work/random_walks.npz --p 1.0 --q 0.5
```
Then, learn embeddings using the random walks
```
cd code #make sure you are in code directory
python train_node2vec.py --graph ../data/co-author-matrix.npz --walk ../work/random_walks.npz --log ../log1/ --save ../results/node_embeddings.npz
```

##Important Notes
Current implementation hard code several parameters:  
The number of dimension d = 200  
The number of epochs = 1  
The number of walk per node r = 1  
Random walk length l = 100  
Context size k = 16  
You should modify these parameters as well as p an q. In particular, r should be increased if you want to use for real application. However, my implementation is not efficient so it takes hours with the example network (74,530 nodes), so I restrict r = 1.   
  
The experimental settings in the original paper are: d=128, epochs=1, r=10, l=80, and k=10.   

## Download random walks and embeddings
I made a sample random walks, and learned embeddings available because it takes time to make them:. You can download them as below:
 - https://googledrive.com/host/0B046sNk0DhCDZ3pla3BKdnllcEE/random_walks.npz
 - https://googledrive.com/host/0B046sNk0DhCDZ3pla3BKdnllcEE/node_embeddings.npz  
 Put random_walks.npz into ./work and node_embeddings.npz into ./results
 
## Vector Examples
The embeddings are learned with p=1.0, q=0.5, d=200, epochs=1, r=1, l=100, and k=16. 

Top 3 cosine similar authors to Jure Leskovec:  
julian mcauley 0.459304  
jon kleinberg 0.438476  
jaewon yang 0.423793  
Top 3 cosine similar authors to Ying Ding:  
xin shuai 0.438184  
jie tang 0.424988  
jerome r busemeyer 0.395817  

*Note that Ying does not publish so many papers on conferences, but I only use top conferences. So the results might not be intuitive. 
See examples codes on ipython notebook: https://github.com/apple2373/node2vec/blob/master/code/embedding_explore.ipynb

##To do list
1. Make the code more flexible using command line arguments (e.g. dimensions of embeddings)
2. Use multi-processing for computing transition probabilities and random walks. 
3. Use asynchronous SGD (currently using Adam SGD with single process).  
PR welcome especially for 2 and 3.  
