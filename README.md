# re_mcl
This is a Python program for Markov Clustering (MCL) that supports not only dense matrices (to be automatically converted to CSR format) but also sparse matrices (can read Matrix Market mtx files). The convergence process can also be reproduced using the logger protocol. Run

    pip install git+https://github.com/hilolani/re_mcl.git

to use this program. (If you are using Google Colab, put "!" before "pip")

Several adjacent matrices for demonstration purposes are stored in this repository as Matrix Market mtx files and can be used for calculations such as MCL as follows. The usage of these basic datasets overlaps with the readme in the MiF repository, so it will be omitted here. Please refer to that document for details.

    from re_mcl import *

    re_mcl= load_adjmats()

    mtxlist = [re_mcl.gadget,re_mcl.karateclub,re_mcl.erdosReny,re_mcl.scalefree,re_mcl.homophilly,re_mcl.heterophilly,re_mcl.eat]

    adjacencyinfocheckedlist = [adjacencyinfocheck(i) for i in mtxlist]

    adjacencylist = ['gadget', 'karateclub', 'erdosReny', 'scalefree', 'homophilly', 'heterophilly']

    mclprocess(re_mcl.karateclub)

In addition to the conventional MCL, Recurrent MCL (RMCL), developed at the former Akama Laboratory at Tokyo Institute of Technology, has been implemented in this repository and can be computed as follows with the new function of rmcl_basic(). BMCL (Branching Markov Clustering) as a part of RMCL (recurrent MCL) algorithm allows us to overcome a weak point--clusters size unbalance--of the ordinary MCL revealed particularly specifically when it is applied to a document or a corpus. BMCL makes it possible to subdivide a large core cluster into appropriately resized sub-graphs. It consists of creating a virtual adjacency relationship among the Markov hard clusters and producing intrinsically informative complex networks.

    cluslist = mclprocess(re_mcl.scalefree, 20)

    result_branching = rmcl_basic(cluslist,adjacencyinfocheckedlist[3],threspruning=1,reverse_process=False) #The core cluster is divided based on the algorithm of of Branching RMCL.

"""

Also good if you are using Google Colab.

    originalpath = "/content/drive/My Drive/Colab Notebooks/scalefree.mtx"

    result_branching = rmcl_basic(cluslist,originalpath,threspruning=2,reverse_process=False)

"""

    result_reverse_branching = rmcl_basic(cluslist,adjacencyinfocheckedlist[3],threspruning=3,reverse_process=True)
    
    #The clusters other than the core one is size-adjusted (appropriately merged) based on the algorithm of Reverse granching RMCL.

# branching_rmcl
This function's specification is as follows.

    branching_rmcl(dic_mclresult, originadj, defaultcorenum=0, threspruning=1.0, reverse_process = False, logger = None):

The first argument, `dic_mclresult`, directly assigns the return value of the `mcl_process()` function.

The second argument, `originadj`, contains the original adjacency matrix provided initially, but it must have been converted to a CSR sparse matrix via `adjacencyinfocheck()`.

The third argument, `defaultcorenum`, and its default value 0 correspond to the number of the selected core cluster within the list of core clusters. A core cluster refers to any cluster whose size exceeds twice the standard deviation of all clusters output by MCL. In other words, if no core cluster exists, the `rmcl_branching()` function terminates with the message “There is no core cluster, so no need to run rmcl.” Typically, there is only one core cluster, numbered 0. However, if there are two or more core cluster candidates, they must be selected from the sorted list of candidates by size, with the second candidate being 1, the third being 2, and so on. This function is designed not to allow the selection and execution of multiple core clusters at once. 

The fourth argument, threspruning, refers to the threshold for latent adjacency weights. Latent adjacency refers to the connection between two representative nodes (the nodes with the highest degree within their respective clusters) that are not directly adjacent, achieved via a two-step path through an intermediate cluster. 

The fifth argument, `reverse_process`, has a default value of `False`. If `False`, it performs standard branching MCL for core cluster partitioning. If `True`, it instead performs reverse branching MCL, which appropriately re-merges clusters other than the core clusters.

# srmcl
Simply-repeated MCL, abbreviated as srmcl, operates as follows: if a core cluster exists, it extracts the hub with the highest degree from within it and then re-runs MCL iteratively only on the core cluster. In the sr_mcl() function, due to its specification, even if multiple core clusters exist, srmcl is applied only to the largest one among them. Clusters other than the core cluster retain the results from the initial MCL. If coreinfoonly = False (default is True), non-core clusters are merged into the srmcl results.

# Ingenuity:
The output of MCL typically takes the form of a Python dictionary, where the key is the cluster number and the value represents the nodes (vertices) belonging to that cluster. However, to simplify calculations, functions such as `mcldict_to_mcllist()` and `mcldict_to_mclset()` are provided to convert this dict-formatted `mclresult` into either nested list format or set format, which can output unions or intersections between clusters.

This MCL program uses CSR (Compressed Sparse Row) format as the default sparse matrix representation, but it may convert to COO (Coordinate) format during computation. Then, assing the original object as it is to the logger would result in massive amounts of Row indices, Column indices, and Values being displayed in the log file and standard output. Therefore, a SafeCSR class is implemented to suppress this inconvenient behavior.

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

#Selecting the representative node for each cluster Ck . 

    Representaive_ClusterStagek = {Max (Degree (Ck(j) ) ) | j=1,2,….,d};

#Removing the representative node of the core cluster from the following two lists which are the arguments of the function Complement,

    CC’ n= Complement(CoreCluster n, Representaive_ClusterStagek) 

    RCS’k= Complement(Representaive_ClusterStagek, CoreCluster n)

#The Function ExtractAdjacency(adjacency_matrix, {row_number,column_number})is to extract rows and columns of an adjacency matrix,

    CC’ n_RCS’k =ExtractAdjacency(G, {CC’ n, RCS’k })

#Tr means transposition of a matrix. 

    PathNumbersMatrix n =CC’ n_RCS’k * Tr(CC’ n_RCS’k);

#Generating an adjacency matrix by setting all diagonal elements=0 and all non diagonal elements larger than 1 = 1.

    LatentAdjacencyMatrix n =MakeAdjacencyMatrix(PathNumbersMatrix n);

#Repeat MCL.

    MCL(LatentAdjacencyMatrix n);


# References

Stijn van Dongen, Graph Clustering by Flow Simulation, 2000
https://dspace.library.uu.nl/bitstream/handle/1874/848/full.pdf?sequence=1&isAllowed=y

Jaeyoung Jung and Hiroyuki Akama. 2008. Employing Latent Adjacency for Appropriate Clusteringof Semantic Networks. New Trends in Psychometrics p.131-140

Hiroyuki Akama et al, 2008. Random graph model simulations of semantic networks for associative Concept dictionaries, TextGraphs-3 doi: https://dl.acm.org/doi/10.5555/1627328.1627337

Hiroyuki Akama et al., 2008. How to Take Advantage of the Limitations with Markov Clustering?--The Foundations of Branching Markov Clustering (BMCL), IJCNLP-2008, p.901~906
https://aclanthology.org/I08-2129.pdf

Hiroyuki Akama et al., 2007. Building a clustered semantic network for an Entire Large Dictionary of Japanese, PACLING-2007, p.308~316
https://www.researchgate.net/publication/228950233_Building_a_clustered_semantic_network_for_an_Entire_Large_Dictionary_of_Japanese

Jaeyoung Jung, Maki Miyake, Hiroyuki Akama. 2006. Recurrent Markov Cluster (RMCL) Algorithm for the Refinement of the Semantic Network. In: LREC. p. 1428–1431 http://www.lrec-conf.org/proceedings/lrec2006/
