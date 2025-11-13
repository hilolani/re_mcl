# re_mcl
This is a Python program for Markov Clustering (MCL) that supports not only dense matrices (to be automatically converted to CSR format) but also sparse matrices (can read Matrix Market mtx files). The convergence process can also be reproduced using the logger protocol. Run

pip install git+https://github.com/hilolani/re_mcl.git

to use this program. 

Several adjacent matrices for demonstration purposes are stored in this repository as Matrix Market mtx files and can be used for calculations such as MCL as follows.

from re_mcl import *

re_mcl= load_adjmats()

mtxlist = [re_mcl.gadget,re_mcl.karateclub,re_mcl.erdosReny,re_mcl.scalefree,re_mcl.homophilly,re_mcl.heterophilly,re_mcl.eat]

adjacencyinfocheckedlist = [adjacencyinfocheck(i) for i in mtxlist]

adjacencylist = ['gadget', 'karateclub', 'erdosReny', 'scalefree', 'homophilly', 'heterophilly']

mclprocess(re_mcl.karateclub)

In addition to the conventional MCL, Recurrent MCL (RMCL), developed at the former Akama Laboratory at Tokyo Institute of Technology, has been implemented in this repository and can be computed as follows with the new function of rmcl_basic().

cluslist = mclprocess(re_mcl.scalefree, 20)

result_branching = rmcl_basic(cluslist,adjacencyinfocheckedlist[3],threspruning=1,reverse=False) #The core cluster is divided based on the algorithm of of Branching RMCL.

"""

Also good if you are using Google Colab.

originalpath = "/content/drive/My Drive/Colab Notebooks/scalefree.mtx"

result_branching = rmcl_basic(cluslist,originalpath,threspruning=2,reverse=False)

"""

result_reverse_branching = rmcl_basic(cluslist,adjacencyinfocheckedlist[3],threspruning=3,reverse=True)#The clusters other than the core one is size-adjusted (appropriately merged) based on the algorithm of Reverse granching RMCL.

# Notes:
The following is the basic algorithm of MCL and Branching MCL (based on the latent adjacency between the core cluster and the others.) The reverse Branching MCL consists of appropriately consolidating other fragmented clusters by swapping the order of the matrices to compute the product for PathNumbersMatrix such as n = Tr(CC’ n_RCS’k) * CC’ n_RCS’k.

# The MCL algorithm 

#The following follows but modifies a little Figure15 that is proposed in Van Dongen’s thesis, p.55.

MCL (G,e,r) { 

    G=G+I; T1=TG;
    
    for k=1,...,∞{
        T2k=Expe(T2k-1); # Expansion 
        T2k+1=Γr(T2k);	# Inflation
    }
    
    #Starting cluster stage. 

    for i=1,...,n {
       T2k+1 = = [tij](i=1,2,…,m; j=1,2,…,m);
       Ci={[tij]| for j=1,...,m{[tij]>0.1};};
    }
    
    #Ending cluster stage.

    ClusterStagek={Ck(1), Ck(2), ..., Ck(d)}; 

    If(T2k+1 is (near-) idempotent) break;
}

# The RMCL algorithms

#Below is the BMCL algorithm to divide large-sized core clusters made by the original MCL.

#Selecting Core Clusters,

if(Size(Ck(p)) > 2*Standard Deviation(Size Ck(j)), then CoreCluster n = Ck(p);

#Selecting the representative node for each cluster Ck . Representaive_ClusterStagek = {Max (Degree (Ck(j) ) ) | j=1,2,….,d};

#Removing the representative node of the core cluster from the following two lists which are the arguments of the function Complement,

CC’ n= Complement(CoreCluster n, Representaive_ClusterStagek) 

RCS’k= Complement(Representaive_ClusterStagek, CoreCluster n)

#The Function ExtractAdjacency(adjacency_matrix, {row_number,column_number})is to extract rows and columns of an adjacency matrix,

CC’ n_RCS’k =ExtractAdjacency(G, {CC’ n, RCS’k })

#Tr means transposition of a matrix. 

PathNumbersMatrix n =CC’ n_RCS’k * Tr(CC’ n_RCS’k);

#Generating an adjacency matrix by setting all diagonal elements=0 and all non diagonal elements larger than 1 = 1.

LatentAdjacencyMatrix n =MakeAdjacencyMatrix(PathNumbersMatrix n);

#Repeat MCL, MCL(LatentAdjacencyMatrix n);


References

Stijn van Dongen, Graph Clustering by Flow Simulation, 2000
https://dspace.library.uu.nl/bitstream/handle/1874/848/full.pdf?sequence=1&isAllowed=y

Jaeyoung Jung and Hiroyuki Akama. 2008. Employing Latent Adjacency for Appropriate Clusteringof Semantic Networks. New Trends in Psychometrics p.131-140

Hiroyuki Akama et al, 2008. Random graph model simulations of semantic networks for associative Concept dictionaries, TextGraphs-3 doi: https://dl.acm.org/doi/10.5555/1627328.1627337

Jaeyoung Jung, Maki Miyake, Hiroyuki Akama. 2006. Recurrent Markov Cluster (RMCL) Algorithm for the Refinement of the Semantic Network. In: LREC. p. 1428–1431 http://www.lrec-conf.org/proceedings/lrec2006/
