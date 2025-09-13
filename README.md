# re_mcl
This is a Python program for Markov Clustering (MCL) that supports not only dense matrices (to be automatically converted to CSR format) but also sparse matrices (can read Matrix Market mtx files). The convergence process can also be reproduced using the logger protocol. Run

pip install git+https://github.com/hilolani/re_mcl.git

to use this program.
We plan to implement Recurrent MCL (RMCL) and Latent adjacency clustering, developed at the former Akama Laboratory at Tokyo Institute of Technology, in this repository going forward.

References


Jaeyoung Jung and Hiroyuki Akama. 2008. Employing Latent Adjacency for Appropriate Clusteringof Semantic Networks. New Trends in Psychometrics p.131-140

Hiroyuki Akama et al, 2008. Random graph model simulations of semantic networks for associative Concept dictionaries, TextGraphs-3 doi: https://dl.acm.org/doi/10.5555/1627328.1627337

Jaeyoung Jung, Maki Miyake, Hiroyuki Akama. 2006. Recurrent Markov Cluster (RMCL) Algorithm for the Refinement of the Semantic Network. In: LREC. p. 1428–1431 http://www.lrec-conf.org/proceedings/lrec2006/
