# re_mcl
This is a Python program for Markov Clustering (MCL) that supports not only dense matrices (to be automatically converted to CSR format) but also sparse matrices (can read Matrix Market mtx files). The convergence process can also be reproduced using the logger protocol. Run

pip install git+https://github.com/hilolani/re_mcl.git

to use this program. 

Several adjacent matrices for demonstration purposes are stored in this repository as Matrix Market mtx files and can be used for calculations such as MCL as follows.

from re_mcl import *

re_mcl= load_adjmats()

mtxlist = [re_mcl.gadget,re_mcl.karateclub,re_mcl.erdosReny,re_mcl.scalefree,re_mcl.homophilly,re_mcl.heterophilly,re_mcl.eat]

mclprocess(re_mcl.karateclub)

In addition to the conventional MCL, Recurrent MCL (RMCL), developed at the former Akama Laboratory at Tokyo Institute of Technology, has been implemented in this repository and can be computed as follows with the new function of rmcl_basic().

cluslist = mclprocess(re_mcl.scalefree, 20)

rmcl_basic(cluslist,re_mcl.scalefree)#The core cluster is divided based on the algorithm of RMCL.

References

Stijn van Dongen, Graph Clustering by Flow Simulation, 2000
https://dspace.library.uu.nl/bitstream/handle/1874/848/full.pdf?sequence=1&isAllowed=y

Jaeyoung Jung and Hiroyuki Akama. 2008. Employing Latent Adjacency for Appropriate Clusteringof Semantic Networks. New Trends in Psychometrics p.131-140

Hiroyuki Akama et al, 2008. Random graph model simulations of semantic networks for associative Concept dictionaries, TextGraphs-3 doi: https://dl.acm.org/doi/10.5555/1627328.1627337

Jaeyoung Jung, Maki Miyake, Hiroyuki Akama. 2006. Recurrent Markov Cluster (RMCL) Algorithm for the Refinement of the Semantic Network. In: LREC. p. 1428–1431 http://www.lrec-conf.org/proceedings/lrec2006/
