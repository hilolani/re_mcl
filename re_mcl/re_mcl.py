# -*- coding: utf-8 -*-
import numpy as np
import os
import networkx as nx
import math
import itertools
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix,coo_matrix,csc_matrix,issparse,isspmatrix_csr,isspmatrix_csc,isspmatrix_coo,csgraph
from collections import defaultdict
from sklearn.utils import Bunch
try:
    # Python 3.9+
    from importlib.resources import files
except ImportError:
    # Python 3.7–3.8
    from importlib_resources import files
from . import data  # re_mcl/data/
import shutil
from math import isclose
import logging

formatter = logging.Formatter("%(asctime)s [MCL] %(message)s", "%Y-%m-%d %H:%M:%S")
result_handler = logging.FileHandler("mcl_results.log", mode="w", encoding="utf-8")
result_handler.setFormatter(formatter)
mcl_logger = logging.getLogger("mcl_results")
mcl_logger.setLevel(logging.INFO)
mcl_logger.addHandler(result_handler)
mcl_logger.propagate = False

#If you use a Google Colab user, run the following.
#from google.colab import drive
#drive.mount('/content/drive')
#from google.colab import files

def fileOnColab(filename, basepath = "/content/drive/My Drive/Colab Notebooks"):
    filepath = os.path.join(basepath, filename)
    print(filepath)
    return filepath

def load_adjmats(return_X_y=False, as_frame=False, scaled=False):
    base_path = os.path.join(os.path.dirname(__file__), "data")    
    return Bunch(
        erdosReny = os.path.join(base_path, "ErdosReny.mtx"),
        gadget = os.path.join(base_path, "gadget.mtx"),
        heterophilly = os.path.join(base_path, "heterophilly.mtx"),
        homophilly = os.path.join(base_path, "homophilly.mtx"),
        karateclub = os.path.join(base_path, "karateclub.mtx"),
        scalefree = os.path.join(base_path, "scalefree.mtx"),
        DESCR="This is a toy dataset consisting of six sparse matrices in Matrix Market format."
    )

def adjanceyinfocheck(adjacencymatrix):
    if isinstance(adjacencymatrix, np.ndarray):
        print("The graph is given as a dense matrix.")
        adj_matrix = csr_matrix(adjacencymatrix)
        return adj_matrix
    elif isspmatrix_csr(adjacencymatrix):
        #print("The graph is given as a sparse matrix with the csr format.")
        adj_matrix = adjacencymatrix
        return adj_matrix
    elif adjacencymatrix.endswith(".mtx"):
        print("The graph is given under the format of MatrixMarket with 0-based indexes.")
        adj_matrix = mmread(adjacencymatrix).tocsr()
        return adj_matrix
    else:
        raise ValueError("Unsupported format or indexing. This function is designed to work with sparse matrix files created in languages that use 0-based indexing. If there is something wrong, check the indexing of your sparse matrix.")

def prepro(adjancencymatrix):
  adj_matrix = adjanceyinfocheck(adjancencymatrix)
  adj_matrix_original = adj_matrix.copy()
  adj_matrix = adj_matrix.transpose()
  adj_matrix = adj_matrix + csr_matrix(np.eye(np.shape(adj_matrix)[0]))
  return adj_matrix

def rescaling(adjancencymatrix):
   #adj_matrix = transition(prepro(mif.karateclub))
   adj_matrix = adjancencymatrix.copy()
   col_sums = np.array(adj_matrix.sum(axis=0)).ravel()
   col_sums[col_sums == 0] = 1
   inv_col_sums = 1.0 / col_sums
   scaling_matrix = csc_matrix(np.diag(inv_col_sums))
   return adj_matrix @ scaling_matrix

def expansion(adjancencymatrix):
  #adj_matrix = prepro(adjancencymatrix)
  adj_matrix = adjancencymatrix.copy()
  expanded_adj_matrix = adj_matrix @ adj_matrix
  return expanded_adj_matrix

def hadamardpower(adjancencymatrix):
  #adj_matrix = transition(prepro(mif.karateclub))
  adj_matrix = adjancencymatrix.copy()
  hd = np.square(adj_matrix.data)
  adj_matrix.data = hd
  return adj_matrix

def inflation(adjancencymatrix):
  #adj_matrix = transition(prepro(mif.karateclub))
  adj_matrix = adjancencymatrix.copy()
  hadamarded = hadamardpower(adj_matrix)
  inflated = rescaling(hadamarded)
  return inflated

def normalizedq(adjacencymatrix):
    adj_matrix = adjacencymatrix.copy()
    col_sums = np.array(adj_matrix.sum(axis=0)).ravel()
    if np.allclose(col_sums, 1.0, atol=1e-12, rtol=0.0):
        print("Rescaled.")
        result = True
    else:
        print("Not rescaled.")
        result = False
    return result

def get_soft_clusters_proto(adjacencymatrix, threshold=1e-6, eps=1e-12):
    adj_matrix = adjacencymatrix.copy()
    adj_matrix_csc = adj_matrix if isinstance(adj_matrix, csc_matrix) else adj_matrix.tocsc(copy=False)
    col_sums = np.array(adj_matrix_csc.sum(axis=0)).ravel()
    zero_rows =np.where(adj_matrix_csc.getnnz(axis=1) == 0)[0]
    num = list(range(adj_matrix.shape[0] - len(zero_rows)))
    adj_matrix_csr = adj_matrix if isinstance(adj_matrix, csr_matrix) else adj_matrix.tocsr(copy=False)
    diag_vals = adj_matrix.diagonal()
    seed_rows = np.where(diag_vals > threshold)[0]
    clusters = []
    for i in seed_rows:
        start, end = adj_matrix_csr.indptr[i], adj_matrix_csr.indptr[i+1]
        cols_i = adj_matrix_csr.indices[start:end]
        vals_i = adj_matrix_csr.data[start:end]
        members = cols_i[vals_i > threshold]
        flag = np.zeros(adj_matrix_csr.shape[0], dtype=float)
        flag[members] = 1.0
        clusters.append(flag)
    if len(clusters) == 0:
        print("No seed rows over threshold. (No clusters at this step.)")
        return {}, {}
    flags = np.vstack(clusters)
    overlapinfo = flags.sum(axis=0)
    if overlapinfo.max() > 1 + eps:
        print("Soft clusters are recorded at this step. Continue.")
        convergence = False
    else:
        print("Hard clusters are gotten at this step (no overlaps). Stop.")
        convergence = True
    clusinfo = {k: flags[k, :].tolist() for k in range(flags.shape[0])}
    clustersatthisstep = {k: np.where(flags[k, :] > 0.5)[0].tolist()
                          for k in range(flags.shape[0])}
    mcl_logger.info('Convergence:{}'.format(convergence))
    mcl_logger.info('Cluster Matrix:{}'.format(clusinfo))
    mcl_logger.info('Cluster list:{}'.format(clustersatthisstep))
    return convergence,clusinfo, clustersatthisstep

def mclprocess(adjancencymatrix, stepnum = 10):
    steps = 0
    stmat = inflation(expansion(rescaling(prepro(adjancencymatrix))))
    rwmat = stmat
    while steps < stepnum:
        convergence, clustersatthisstep = get_soft_clusters_proto(rwmat)
        if convergence == False:
            rwmat = inflation(expansion(rwmat))
        else:
            print('Convergence:{}'.format(convergence))
            print('Cluster list:{}'.format(clustersatthisstep))
            print('To get more information, run "!cat mcl_results.log"')
            return clustersatthisstep
            break
