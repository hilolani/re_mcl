# -*- coding: utf-8 -*-
import numpy as np
import os
import networkx as nx
import math
import itertools
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix,coo_matrix,csc_matrix,issparse,isspmatrix_csr,isspmatrix_csc,isspmatrix_coo,csgraph,lil_matrix,dok_matrix,dia_matrix,load_npz
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
import io
import json
import pickle

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
        eat =  os.path.join(base_path, "eat.mtx"),
        DESCR="This is a toy dataset consisting of six sparse matrices in Matrix Market format."
    )

def adjacencyinfocheck_old(adjacencymatrix):
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

def adjacencyinfocheck(adjacencymatrix, logger=None):
  path_or_matrix = adjacencymatrix
  if isinstance(path_or_matrix, str) and os.path.exists(path_or_matrix):
        path = path_or_matrix
        ext = os.path.splitext(path)[1].lower()
        if logger:
            logger.info(f"Loading matrix from file: {path}")
        if ext == ".mtx":
            matrix = mmread(path).tocsr()
            if logger: logger.info("Loaded .mtx file → CSR.")
            return matrix
        elif ext == ".npz":
            loaded = np.load(path, allow_pickle=True)
            if 'data' in loaded and 'indices' in loaded and 'indptr' in loaded:
                matrix = load_npz(path).tocsr()
                if logger: logger.info("Loaded sparse .npz file → CSR.")
                return matrix
            else:
                matrix = csr_matrix(loaded['arr_0'])
                if logger: logger.info("Loaded dense .npz file → CSR.")
                return matrix
        elif ext == ".pkl":
            with open(path, "rb") as f:
                obj = pickle.load(f)
                return adjacencyinfocheck(obj, logger=logger)
        elif ext == ".csv":
            arr = np.loadtxt(path, delimiter=",")
            matrix = csr_matrix(arr)
            if logger: logger.info("Loaded .csv file → CSR.")
            return matrix
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if all(k in data for k in ("row", "col", "data", "shape")):
                row = np.array(data["row"], dtype=np.int64)
                col = np.array(data["col"], dtype=np.int64)
                vals = np.array(data["data"], dtype=np.float64)
                shape = tuple(data["shape"])
                matrix = coo_matrix((vals, (row, col)), shape=shape).tocsr()
                if logger: logger.info("Loaded sparse JSON (COO format) → CSR.")
                return matrix
            else:
                arr = np.array(data, dtype=np.float64)
                matrix = csr_matrix(arr)
                if logger: logger.info("Loaded dense JSON → CSR.")
                return matrix
        else:
            msg = f"Unsupported file extension: {ext}"
            if logger: logger.error(msg)
            raise ValueError(msg)
  matrix = path_or_matrix
  if isinstance(matrix, csr_matrix):
      if logger: logger.info("Matrix is already CSR format.")
      return matrix
  elif issparse(matrix):
      if logger: logger.info(f"Converting {type(matrix).__name__} → CSR.")
      return matrix.tocsr()
  elif isinstance(matrix, np.ndarray):
      if logger: logger.info("Converting ndarray → CSR.")
      return csr_matrix(matrix)
  else:
      msg = f"Unsupported input type: {type(matrix)}"
      if logger: logger.error(msg)
      raise TypeError(msg)
      
def prepro(adjacencymatrix):
  adj_matrix = adjacencyinfocheck(adjacencymatrix)
  adj_matrix_original = adj_matrix.copy()
  adj_matrix = adj_matrix.transpose()
  adj_matrix = adj_matrix + csr_matrix(np.eye(np.shape(adj_matrix)[0]))
  return adj_matrix

def rescaling(adjacencymatrix):
   #adj_matrix = transition(prepro(mif.karateclub))
   adj_matrix = adjacencymatrix.copy()
   col_sums = np.array(adj_matrix.sum(axis=0)).ravel()
   col_sums[col_sums == 0] = 1
   inv_col_sums = 1.0 / col_sums
   scaling_matrix = csc_matrix(np.diag(inv_col_sums))
   return adj_matrix @ scaling_matrix

def expansion(adjacencymatrix):
  #adj_matrix = prepro(adjacencymatrix)
  adj_matrix = adjacencymatrix.copy()
  expanded_adj_matrix = adj_matrix @ adj_matrix
  return expanded_adj_matrix

def hadamardpower(adjacencymatrix):
  #adj_matrix = transition(prepro(mif.karateclub))
  adj_matrix = adjacencymatrix.copy()
  hd = np.square(adj_matrix.data)
  adj_matrix.data = hd
  return adj_matrix

def inflation(adjacencymatrix):
  #adj_matrix = transition(prepro(mif.karateclub))
  adj_matrix = adjacencymatrix.copy()
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

def mclprocess(adjacencymatrix, stepnum = 20):
    steps = 0
    stmat = inflation(expansion(rescaling(prepro(adjacencymatrix))))
    rwmat = stmat
    while steps < stepnum:
        convergence, clusinfo, clustersatthisstep = get_soft_clusters_proto(rwmat)
        if convergence == False:
            rwmat = inflation(expansion(rwmat))
        else:
            print('Convergence:{}'.format(convergence))
            print('Cluster list:{}'.format(clustersatthisstep))
            print('To get more information, run "!cat mcl_results.log"')
            return clustersatthisstep
            break
        steps = steps + 1

def rmcl_basic(dic_mclresult, mtx_originadj, defaultcorenum=0, threspruning=1.0):
    originadj =  mtx_originadj
    mmoriginadj = mmread(originadj)
    cluslist = dic_mclresult
    clusmemlist=[cluslist[i] for i in range(len(cluslist))]
    clussizelist = [len(j) for j in clusmemlist]
    corecluscandlist = np.where(np.array(clussizelist)>np.mean(clussizelist) + 2*np.std(clussizelist).tolist())[0].tolist()
    if corecluscandlist==[]:
        mcl_logger.info('Warning:{}'.format("There is no core cluster, so no need to run rmcl."))
        print('Warning:{}'.format("There is no core cluster, so no need to run rmcl."))
    else:
        corecluscanddata = list(zip(corecluscandlist,[clussizelist[i] for i in corecluscandlist]))
        defaultcore = defaultcorenum
        coreclusternumber = corecluscanddata[defaultcore][0]
        G = nx.from_scipy_sparse_array(mmoriginadj)
        deginfo = G.degree
        clusmemdeginfo = [[deginfo[i] for i in clusmemlist[j]] for j in range(len(clusmemlist))]
        max_indices = [np.argwhere(i == np.max(i)).flatten().tolist() for i in clusmemdeginfo]
        allrepresentnodeslist = [clusmemlist[i][max_indices[i][0]] for i in range(len(max_indices))]
        allbutcorerepresentnodeslist = [elem for i, elem in enumerate(allrepresentnodeslist) if i !=coreclusternumber]
        coreclustermember = clusmemlist[coreclusternumber]
        coreclusterbutrepresentmember = [elem for i, elem in enumerate(coreclustermember) if elem != allrepresentnodeslist[coreclusternumber]]
        comblist = list(itertools.product(coreclusterbutrepresentmember,allbutcorerepresentnodeslist))
        values = {(i, j): v for i, j, v in zip(mmoriginadj.row, mmoriginadj.col, mmoriginadj.data)}
        results = [values.get(comblist, 0.0) for comblist in comblist]
        coreclusbutrepresentrow = list(range(len(coreclusterbutrepresentmember)))
        allbutcoreclusrepresentcol = [i + coreclusbutrepresentrow[-1] + 1 for i in list(range(len(allbutcorerepresentnodeslist)))]
        corresrow = list(zip(coreclusterbutrepresentmember,coreclusbutrepresentrow))
        correscol = list(zip(allbutcorerepresentnodeslist,allbutcoreclusrepresentcol))
        focusedcomblist = list(itertools.product(coreclusbutrepresentrow, allbutcoreclusrepresentcol))
        rows, cols = zip(*focusedcomblist)
        focusedshape = len(coreclusterbutrepresentmember) + len(allbutcorerepresentnodeslist)
        focuesd_sparse_mat = coo_matrix((results, (rows, cols)), shape=(focusedshape,focusedshape))
        focusedlatentadj = focuesd_sparse_mat*focuesd_sparse_mat.T
        focusedlatentadj = focusedlatentadj.tocoo()
        focusedlatentadj = coo_matrix((focusedlatentadj.data, (focusedlatentadj.row, focusedlatentadj.col)), shape=(len(coreclusterbutrepresentmember),len(coreclusterbutrepresentmember)))
        focusedlatentadj.setdiag(0.0)
        focusedlatentadj.eliminate_zeros()
        tmpla = focusedlatentadj.copy()
        thresp = threspruning
        tmpla.data = np.where(tmpla.data < thresp, 0.0, 1.0)
        tmpla.eliminate_zeros()
        tmplacsr=tmpla.tocsr()
        mcl_logger.info('RMCL:{}'.format("RMCL will start soon."))
        rmclresult = mclprocess(tmplacsr)
        print(f"rmcresult:{rmclresult}")
        if rmclresult==None:
            mcl_logger.info('Warning:{}'.format("RMCL not possible."))
            print('Warning:{}'.format("RMCL not possible."))
        else:
            corecluscorespond = list(zip(coreclusterbutrepresentmember,coreclusbutrepresentrow))
            mapping = {v: k for k, v in dict(corecluscorespond).items()}
            finalrmclresult = {k: [mapping[x] for x in v if x in mapping] for k, v in rmclresult.items()}
            mcl_logger.info('Final result of rmcl after renumbering--core reclustering:{}'.format(finalrmclresult))
            print(f"Final result of rmcl after renumbering--core reclustering: {finalrmclresult}")
