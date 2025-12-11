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
import sys
from typing import Any, Tuple, Dict, List, Optional

#If you use a Google Colab user, run the following.
#from google.colab import drive
#drive.mount('/content/drive')
#from google.colab import files

def fileOnColab(filename, basepath = "/content/drive/My Drive/Colab Notebooks"):
    filepath = os.path.join(basepath, filename)
    print(filepath)
    return filepath

formatter = logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")
logger_a = logging.getLogger("MiF")
logger_a.setLevel(logging.INFO)

fh_a = logging.FileHandler("MiF.log", mode="w", encoding="utf-8")
fh_a.setFormatter(formatter)
ch_a = logging.StreamHandler(sys.stdout)
ch_a.setFormatter(formatter)

logger_a.addHandler(fh_a)
logger_a.addHandler(ch_a)
logger_a.propagate = False

logger_b = logging.getLogger("MatrixLoader")
logger_b.setLevel(logging.INFO)

fh_b = logging.FileHandler("matrix.log", mode="w", encoding="utf-8")
fh_b.setFormatter(formatter)
ch_b = logging.StreamHandler(sys.stdout)
ch_b.setFormatter(formatter)

logger_b.addHandler(fh_b)
logger_b.addHandler(ch_b)
logger_b.propagate = False

logger_c = logging.getLogger("re_mcl")
logger_c.setLevel(logging.INFO)

fh_c = logging.FileHandler("re_mcl.log", mode="w", encoding="utf-8")
fh_c.setFormatter(formatter)
ch_c = logging.StreamHandler(sys.stdout)
ch_c.setFormatter(formatter)

logger_c.addHandler(fh_c)
logger_c.addHandler(ch_c)
logger_c.propagate = False

def resolve_logger(logger: Optional[logging.Logger], context: str) -> logging.Logger:
    return logger if logger is not None else get_logger(context)

def get_logger(context: str) -> logging.Logger:
    context = context.lower()
    if context in ("mif", "mifdi", "distance", "similarity"):
        return logger_a
    elif context in ("matrix", "loader", "io"):
        return logger_b
    elif context in ("mcl", "re_mcl", "rmcl"):
        return logger_c    
    else:
        raise ValueError(f"Unknown logging context: {context}")

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

def load_mif(*args, **kwargs):
    return load_adjmats(*args, **kwargs)

def load_mcl(*args, **kwargs):
    return load_adjmats(*args, **kwargs)

class SafeCSR(csr_matrix):
    def __repr__(self):
        return f"<SafeCSR shape={self.shape}, nnz={self.nnz}, dtype={self.dtype}>"

    __str__ = __repr__ 

def save_safe_csr_to_mtx(safecsrmatrix, path: str, logger=None):
    log = logger or logging.getLogger(__name__)
    if hasattr(safecsrmatrix, "_csr"):
        safecsrmatrix = safecsrmatrix._csr
    if not isinstance(safecsrmatrix, csr_matrix):
        raise TypeError(f"Expected csr_matrix or SafeCSR, got {type(matrix).__name__}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mmwrite(path, safecsrmatrix)
    log.info(f"Saved CSR matrix to {path}")

def adjacencyinfocheck(adjacencymatrix, logger = None):
    log = resolve_logger(logger, "matrix")
    print(f"log name: {log.name}")
    path_or_matrix = adjacencymatrix
    src = path_or_matrix if isinstance(path_or_matrix, str) else "<in-memory>"
    
    if isinstance(path_or_matrix, str) and os.path.exists(path_or_matrix):
        path = path_or_matrix
        ext = os.path.splitext(path)[1].lower()

        if ext == ".mtx":
            matrix = mmread(path).tocsr()
            log.info("Loaded .mtx file → CSR.")

        elif ext == ".npz":
            loaded = np.load(path, allow_pickle=True)
            if 'data' in loaded and 'indices' in loaded and 'indptr' in loaded:
                matrix = load_npz(path).tocsr()
                log.info("Loaded sparse .npz file → CSR.") 
            else:
                matrix = csr_matrix(loaded['arr_0'])
                log.info("Loaded dense .npz file → CSR.")
                
        elif ext == ".pkl":
            with open(path, "rb") as f:
                obj = pickle.load(f)
                matrix =adjacencyinfocheck(obj)
                
        elif ext == ".csv":
            arr = np.loadtxt(path, delimiter=",")
            matrix = csr_matrix(arr)
            log.info("Loaded .csv file → CSR.")
            
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if all(k in data for k in ("row", "col", "data", "shape")):
                row = np.array(data["row"], dtype=np.int64)
                col = np.array(data["col"], dtype=np.int64)
                vals = np.array(data["data"], dtype=np.float64)
                shape = tuple(data["shape"])
                matrix = coo_matrix((vals, (row, col)), shape=shape).tocsr()
                log.info("Loaded sparse JSON (COO format) → CSR.")
            else:
                arr = np.array(data, dtype=np.float64)
                matrix = csr_matrix(arr)
                log.info("Loaded dense JSON → CSR.")
                
        else:
            msg = f"Unsupported file extension: {ext}"
            log.error(msg)
            raise ValueError(msg)

    else:
        matrix = path_or_matrix

        if isinstance(matrix, csr_matrix):
          log.info(f"Matrix is already CSR format (shape={matrix.shape}, nnz={matrix.nnz})")
          
        elif issparse(matrix):
          matrix = csr_matrix(matrix)  
          log.info(f"Converting {type(matrix).__name__} to CSR format (shape={matrix.shape}, nnz={matrix.nnz})")
         
        elif isinstance(matrix, np.ndarray):
          matrix = csr_matrix(matrix)    
          log.info(f"Converting dense ndarray to CSR format (shape={matrix.shape})")
          
        else:
          msg = f"Unsupported input type: {type(matrix)}"
          log.error(msg)
          raise TypeError(msg)

    log.info(f"Matrix loaded successfully (type={type(matrix).__name__}, shape={matrix.shape}, nnz={matrix.nnz})") 
    return SafeCSR(matrix)
      
def prepro(adjacencymatrixchecked, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    adj_matrix = adjacencymatrixchecked
    adj_matrix_original = adj_matrix.copy()
    adj_matrix = adj_matrix.transpose()
    adj_matrix = adj_matrix + csr_matrix(np.eye(np.shape(adj_matrix)[0]))
    log.info(f"Preprocessing--transposing and adding self-loop-- done.")
    return adj_matrix

def rescaling(adjacencymatrixchecked, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    adj_matrix = adjacencymatrixchecked.copy()
    col_sums = np.array(adj_matrix.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1
    inv_col_sums = 1.0 / col_sums
    scaling_matrix = csc_matrix(np.diag(inv_col_sums))
    log.info(f"rescaling done.")
    return adj_matrix @ scaling_matrix

def expansion(adjacencymatrixchecked, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    adj_matrix = adjacencymatrixchecked.copy()
    expanded_adj_matrix = adjacencymatrixchecked @ adj_matrix
    log.info(f"expansion done.")
    return expanded_adj_matrix

def hadamardpower(adjacencymatrixchecked, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    adj_matrix = adjacencymatrixchecked.copy()
    hd = np.square(adjacencymatrixchecked.data)
    adj_matrix.data = hd
    log.info(f"hadamard power computed.")
    return adj_matrix

def inflation(adjacencymatrixchecked, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    adj_matrix = adjacencymatrixchecked.copy()
    hadamarded = hadamardpower(adj_matrix)
    inflated = rescaling(hadamarded)
    log.info(f"inflation done.")
    return inflated

def normalizedq(adjacencymatrixchecked, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    adj_matrix =adjacencymatrixchecked.copy()
    col_sums = np.array(adj_matrix.sum(axis=0)).ravel()
    if np.allclose(col_sums, 1.0, atol=1e-12, rtol=0.0):
        log.info(f"Rescaled.")
        result = True
    else:
        log.info(f"Not rescaled.")
        result = False
    return result

def get_soft_clusters_proto(adjacencymatrixchecked, threshold=1e-6, eps=1e-12, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    adj_matrix = adjacencymatrixchecked.copy()
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
        log.info(f"No seed rows over threshold. (No clusters at this step.)")
        return {}, {}
    flags = np.vstack(clusters)
    overlapinfo = flags.sum(axis=0)
    if overlapinfo.max() > 1 + eps:
        log.info(f"Soft clusters are recorded at this step. Continue.")
        convergence = False
    else:
        log.info(f"Hard clusters are gotten at this step (no overlaps). Stop.")
        convergence = True
    clusinfo = {k: flags[k, :].tolist() for k in range(flags.shape[0])}
    clustersatthisstep = {k: np.where(flags[k, :] > 0.5)[0].tolist()
                          for k in range(flags.shape[0])}
    log.info(f"Convergence: {convergence}")
    #log.info(f"Cluster Matrix: {clusinfo}")
    log.info(f"Cluster list: {clustersatthisstep}")
    return convergence,clusinfo, clustersatthisstep

def mclprocess(adjacencymatrixchecked, stepnum = 20, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    steps = 0
    stmat = inflation(expansion(rescaling(prepro(adjacencymatrixchecked))))
    rwmat = stmat
    while steps < stepnum:
        convergence, clusinfo, clustersatthisstep = get_soft_clusters_proto(rwmat)
        if convergence == False:
            rwmat = inflation(expansion(rwmat))
        else:
            log.info(f"Convergence: {convergence}")
            log.info(f"Cluster list: {clustersatthisstep}")
            return clustersatthisstep
            break
        steps = steps + 1

def coreclusQ(dic_mclresult_, originadj_, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    if isinstance(originadj_, str) and os.path.exists(originadj_):
        mmoriginadj = mmread(originadj_)
        log.info("The original adjacency information was given as an mtx file.")
    elif isinstance(originadj_, SafeCSR):
        log.info("The original adjacency information was given as a SafeCSR object.")
        pathtmp = os.getcwd() + "/" + "tmp.mtx"
        save_safe_csr_to_mtx(originadj_, pathtmp)
        mmoriginadj = mmread(pathtmp)
    else:
        msg = f"Unsupported file or variable type: {originadj_}"
        log.error(msg)
        raise ValueError(msg)
    clusmemlist=[dic_mclresult_[i] for i in range(len(dic_mclresult_))]
    clussizelist = [len(j) for j in clusmemlist]
    corecluscandlist = np.where(np.array(clussizelist)>np.mean(clussizelist) + 2*np.std(clussizelist).tolist())[0].tolist()
    if corecluscandlist==[]:
        msg = f"Warning: There is no core cluster, so no need to run futher rmcl."
        log.error(msg)
        raise TypeError(msg)
    else:
        log.info(f"There is a core cluster in this MCL result: {corecluscandlist}.")
        return mmoriginadj, clusmemlist, clussizelist, corecluscandlist

def mclus_anaysis(mmoriginadj_, clusmemlist_, clussizelist_, corecluscandlist_, defaultcorenum = 0, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    corecluscanddata_tmp = list(zip(corecluscandlist_,[clussizelist_[i] for i in corecluscandlist_]))
    log.info(f"The candidate(s) of the core cluster are: {corecluscanddata_tmp}")
    corecluscanddata = sorted(corecluscanddata_tmp, key=lambda x: x[1], reverse=True)
    log.info(f"The sorted candidate(s) of the core cluster are: {corecluscanddata}")
    if len(corecluscanddata) >= defaultcorenum + 1:
        coreclusternumber = corecluscanddata[defaultcorenum][0]
    else:
        msg = f"The number (other than 0) that the user assigned to the corecluster selected from the candidates, i.e. {defaultcorenum}, exceeds the extent of the possible corecluster set {len(corecluscanddata)}."
        log.error(msg)
        raise TypeError(msg)
    log.info(f"Due to specification constraints, only one core cluster candidate (with the maximum number of members, if the default setting 0 was kept) is selected. The cluster number is as follows.: {coreclusternumber}")
    G = nx.from_scipy_sparse_array(mmoriginadj_)
    deginfo = G.degree
    clusmemdeginfo = [[deginfo[i] for i in clusmemlist_[j]] for j in range(len(clusmemlist_))]
    max_indices = [np.argwhere(i == np.max(i)).flatten().tolist() for i in clusmemdeginfo]
    allrepresentnodeslist = [clusmemlist_[i][max_indices[i][0]] for i in range(len(max_indices))]
    coreclustermember = clusmemlist_[coreclusternumber]
    log.info(f"The members of the core cluster are:{coreclustermember}")
    noncoreclusternumber = [i for i in range(len(clusmemlist_)) if i not in [coreclusternumber]]
    log.info(f"The numbering of the noncore clusters, noncoreclusternumber is:{noncoreclusternumber}")
    noncoreclustermember = [clusmemlist_[i] for i in noncoreclusternumber]
    log.info(f"The members of the noncore cluster, noncoreclustermember are: {noncoreclustermember}")
    coreclushub = allrepresentnodeslist[coreclusternumber]
    log.info(f"The hub of the core cluster is: {coreclushub}")
    values = {(i, j): v for i, j, v in zip(mmoriginadj_.row, mmoriginadj_.col, mmoriginadj_.data)}
    values_list = {tuple(int(x) for x in k): float(v) for k, v in values.items()}
    log.info(f"All the adjacency information of the original network is: {values_list}")
    coreclusrow = list(range(len(coreclustermember)))
    coreclus_elem_set = set(coreclustermember)
    coreclus_values = {k: v for k, v in values.items() if k[0] in coreclus_elem_set and k[1] in coreclus_elem_set}
    coreclus_values_list = {tuple(int(x) for x in k): float(v) for k, v in coreclus_values.items()}
    log.info(f"The values of the core cluster's inner connection_values: {coreclus_values_list}")
    corecluscorespond = list(zip(coreclustermember,coreclusrow))
    coremapping = {v: k for k, v in dict(corecluscorespond).items()}
    reverse_coremapping = {v: k for k, v in coremapping.items()}
    log.info(f"The mapping dictionary for the core cluster with the number sequence as keys and the original numbers as values: {coremapping}")
    log.info(f"The reverse_coremapping with the original numbers as keys and the number sequence as values: {reverse_coremapping}")
    return coreclushub, coreclusternumber, coreclus_values, coremapping, reverse_coremapping, deginfo, clusmemdeginfo, max_indices, allrepresentnodeslist, coreclustermember, values, noncoreclusternumber, noncoreclustermember
  

def rmcl_branching(dic_mclresult, originadj, defaultcorenum=0, threspruning=1.0, reverse_process = False, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    log.info(f"A little obsolete. Please use the branching_rmcl() function now.")
    if isinstance(originadj, str) and os.path.exists(originadj):
        mmoriginadj = mmread(originadj)
        log.info("The original adjacency information was given as an mtx file.")
    elif isinstance(originadj, SafeCSR):
        log.info("The original adjacency information was given as a SafeCSR object.")
        pathtmp = os.getcwd() + "/" + "tmp.mtx"
        save_safe_csr_to_mtx(originadj, pathtmp)
        mmoriginadj = mmread(pathtmp)
    else:
        msg = f"Unsupported file or variable type: {originadj}"
        log.error(msg)
        raise ValueError(msg)
    clusmemlist=[dic_mclresult[i] for i in range(len(dic_mclresult))]
    clussizelist = [len(j) for j in clusmemlist]
    corecluscandlist = np.where(np.array(clussizelist)>np.mean(clussizelist) + 2*np.std(clussizelist).tolist())[0].tolist()
    if corecluscandlist==[]:
        msg = f"Warning: There is no core cluster, so no need to run rmcl."
        log.error(msg)
        raise TypeError(msg)   
    else:
        corecluscanddata_tmp = list(zip(corecluscandlist,[clussizelist[i] for i in corecluscandlist]))
        log.info(f"The candidate(s) of the core cluster are: {corecluscanddata_tmp}")
        corecluscanddata = sorted(corecluscanddata_tmp, key=lambda x: x[1], reverse=True)
        log.info(f"The sorted candidate(s) of the core cluster are: {corecluscanddata}")
        if len(corecluscanddata) >= defaultcorenum + 1:
            coreclusternumber = corecluscanddata[defaultcorenum][0]
        else:
            msg = f"The number (other than 0) that the user assigned to the corecluster selected from the candidates, i.e. {defaultcorenum}, exceeds the extent of the possible corecluster set {len(corecluscanddata)}."
            log.error(msg)
            raise TypeError(msg)   
        log.info(f"Due to specification constraints, only one core cluster candidate (with the maximum number of members, if the default setting 0 was kept) is selected. The cluster number is as follows.: {coreclusternumber}")
        G = nx.from_scipy_sparse_array(mmoriginadj)
        deginfo = G.degree
        clusmemdeginfo = [[deginfo[i] for i in clusmemlist[j]] for j in range(len(clusmemlist))]
        max_indices = [np.argwhere(i == np.max(i)).flatten().tolist() for i in clusmemdeginfo]
        allrepresentnodeslist = [clusmemlist[i][max_indices[i][0]] for i in range(len(max_indices))]
        allbutcorerepresentnodeslist = [elem for i, elem in enumerate(allrepresentnodeslist) if i !=coreclusternumber]
        coreclustermember = clusmemlist[coreclusternumber]
        coreclusterbutrepresentmember = [elem for i, elem in enumerate(coreclustermember) if elem != allrepresentnodeslist[coreclusternumber]]
        values = {(i, j): v for i, j, v in zip(mmoriginadj.row, mmoriginadj.col, mmoriginadj.data)}
        coreclusbutrepresentrow = list(range(len(coreclusterbutrepresentmember)))
        allbutcoreclusrepresentcol = [i + coreclusbutrepresentrow[-1] + 1 for i in list(range(len(allbutcorerepresentnodeslist)))]
        corresrow = list(zip(coreclusterbutrepresentmember,coreclusbutrepresentrow))
        correscol = list(zip(allbutcorerepresentnodeslist,allbutcoreclusrepresentcol))
        corecluscorespond = list(zip(coreclusterbutrepresentmember,coreclusbutrepresentrow))
        coremapping = {v: k for k, v in dict(corecluscorespond).items()}
        noncoreclusbutrepresentrow = list(range(len(allbutcorerepresentnodeslist)))
        noncorecluscorespond = list(zip(allbutcorerepresentnodeslist,noncoreclusbutrepresentrow))
        noncoremapping = {v: k for k, v in dict(noncorecluscorespond).items()}
        comblist = list(itertools.product(coreclusterbutrepresentmember,allbutcorerepresentnodeslist))
        results = [values.get(comblist, 0.0) for comblist in comblist]
        focusedcomblist = list(itertools.product(coreclusbutrepresentrow, allbutcoreclusrepresentcol))
        rows, cols = zip(*focusedcomblist)
        cols = tuple([x - allbutcoreclusrepresentcol[0] for x in cols])
        focused_sparse_mat = coo_matrix((results, (rows, cols)), shape=(len(coreclusterbutrepresentmember),len(allbutcorerepresentnodeslist)))
        log.info(f"focused_sparse_mat size is: {focused_sparse_mat.shape[0]}, {focused_sparse_mat.shape[1]}")
        focused_sparse_mat_csr = focused_sparse_mat.tocsr()
        if reverse_process == False:
            log.info("reverse_process: False. We are running branching MCL.")
            algorithm = "branching mcl as core reclustering"
            focusedlatentadj=focused_sparse_mat_csr @ focused_sparse_mat_csr.T
        elif reverse_process == True:
            log.info("reverse_process: True. We are running reverse branching MCL.")
            algorithm = "reverse branching mcl as non-core reclustering"
            focusedlatentadj=focused_sparse_mat_csr.T @ focused_sparse_mat_csr
        log.info(f"The shape of the rmcl target csr matrix : {focusedlatentadj.shape}.")
        focusedlatentadj.data[focusedlatentadj.data < threspruning] = 0.0
        focusedlatentadj.setdiag(0.0)
        focusedlatentadj.eliminate_zeros()
        log.info("The target CSR matrix for RMCL has been created.")
        log.info(f"RMCL: RMCL will start soon.")
        rmclresult = mclprocess(focusedlatentadj)
        log.info(f"rmcresult: {rmclresult}")
        if rmclresult==None:
            log.info(f"Warning: RMCL not possible.")
        else:
            if reverse_process == False:
                finalrmclresult = {k: [coremapping[x] for x in v if x in coremapping] for k, v in rmclresult.items()}
                finalresulttmp = finalrmclresult.copy()
                finalrmclresult_adjusted_total = append_hub_to_recluscore(finalresulttmp,allrepresentnodeslist[coreclusternumber])
                log.info(f"Final result of rmcl after renumbering--{algorithm}:branching-rmcl result without hub in the core cluster: {finalrmclresult}, branching-rmcl result with the hub behind the queue: {finalrmclresult_adjusted_total}")
                return finalrmclresult_adjusted_total
            elif reverse_process == True:
                tmp_rmclresult = {k: [noncoremapping[x] for x in v if x in noncoremapping] for k, v in rmclresult.items()}
                finalrmclresult = [[clusinfo_from_nodes(dic_mclresult, j)[0] for j in sublist] for sublist in tmp_rmclresult.values()]
                finalrmclresult_adjusted_total_tmp =  [[(i, set().union(*[vals for _, vals in tmp]))] for i, tmp in enumerate(finalrmclresult)]
                finalrmclresult_adjusted_total = {k: sorted(list(v)) for sub in  finalrmclresult_adjusted_total_tmp for k, v in sub}
                log.info(f"Final result of rmcl after renumbering--{algorithm}: reverse branching-rmcl result with only the representative nodes: {finalrmclresult}, reverse branching-rmcl result with all the non core members: {finalrmclresult_adjusted_total}")
                return finalrmclresult_adjusted_total

def branching_rmcl(dic_mclresult, originadj, defaultcorenum=0, threspruning=1.0, reverse_process = False, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    mmoriginadj, clusmemlist, clussizelist, corecluscandlist = coreclusQ(dic_mclresult, originadj)
    _, coreclusternumber, _, _, _, deginfo, clusmemdeginfo, max_indices, allrepresentnodeslist,coreclustermember,values, _, _ = mclus_anaysis(mmoriginadj, clusmemlist, clussizelist, corecluscandlist, defaultcorenum = 0)
    allbutcorerepresentnodeslist = [elem for i, elem in enumerate(allrepresentnodeslist) if i !=coreclusternumber]
    coreclusterbutrepresentmember = [elem for i, elem in enumerate(coreclustermember) if elem != allrepresentnodeslist[coreclusternumber]]
    coreclusbutrepresentrow = list(range(len(coreclusterbutrepresentmember)))
    allbutcoreclusrepresentcol = [i + coreclusbutrepresentrow[-1] + 1 for i in list(range(len(allbutcorerepresentnodeslist)))] 
    coreclusbutrepresntcorespond  = list(zip(coreclusterbutrepresentmember,coreclusbutrepresentrow))
    corebutrepresentmapping = {v: k for k, v in dict(coreclusbutrepresntcorespond).items()}
    noncoreclusbutrepresentrow = list(range(len(allbutcorerepresentnodeslist)))
    noncorecluscorespond = list(zip(allbutcorerepresentnodeslist,noncoreclusbutrepresentrow))
    noncoremapping = {v: k for k, v in dict(noncorecluscorespond).items()}
    comblist = list(itertools.product(coreclusterbutrepresentmember,allbutcorerepresentnodeslist))
    results = [values.get(comblist, 0.0) for comblist in comblist]
    focusedcomblist = list(itertools.product(coreclusbutrepresentrow, allbutcoreclusrepresentcol))
    rows, cols = zip(*focusedcomblist)
    cols = tuple([x - allbutcoreclusrepresentcol[0] for x in cols])
    focused_sparse_mat = coo_matrix((results, (rows, cols)), shape=(len(coreclusterbutrepresentmember),len(allbutcorerepresentnodeslist)))
    log.info(f"focused_sparse_mat size is: {focused_sparse_mat.shape[0]}, {focused_sparse_mat.shape[1]}")
    focused_sparse_mat_csr = focused_sparse_mat.tocsr()
    if reverse_process == False:
        log.info("reverse_process: False. We are running branching MCL.")
        algorithm = "branching mcl as core reclustering"
        focusedlatentadj=focused_sparse_mat_csr @ focused_sparse_mat_csr.T
    elif reverse_process == True:
            log.info("reverse_process: True. We are running reverse branching MCL.")
            algorithm = "reverse branching mcl as non-core reclustering"
            focusedlatentadj=focused_sparse_mat_csr.T @ focused_sparse_mat_csr
    log.info(f"The shape of the rmcl target csr matrix : {focusedlatentadj.shape}.")
    focusedlatentadj.data[focusedlatentadj.data < threspruning] = 0.0
    focusedlatentadj.setdiag(0.0)
    focusedlatentadj.eliminate_zeros()
    log.info("The target CSR matrix for RMCL has been created.")
    log.info(f"RMCL: RMCL will start soon.")
    rmclresult = mclprocess(focusedlatentadj)
    log.info(f"rmcresult: {rmclresult}")
    if rmclresult==None:
        log.info(f"Warning: RMCL not possible.")
    else:
        if reverse_process == False:
           finalrmclresult = {k: [corebutrepresentmapping[x] for x in v if x in corebutrepresentmapping] for k, v in rmclresult.items()}
           finalresulttmp = finalrmclresult.copy()
           finalrmclresult_adjusted_total = append_hub_to_recluscore(finalresulttmp,allrepresentnodeslist[coreclusternumber])
           log.info(f"Final result of rmcl after renumbering--{algorithm}:branching-rmcl result without hub in the core cluster: {finalrmclresult}, branching-rmcl result with the hub behind the queue: {finalrmclresult_adjusted_total}")
           return finalrmclresult_adjusted_total
        elif reverse_process == True:
           tmp_rmclresult = {k: [noncoremapping[x] for x in v if x in noncoremapping] for k, v in rmclresult.items()}
           finalrmclresult = [[clusinfo_from_nodes(dic_mclresult, j)[0] for j in sublist] for sublist in tmp_rmclresult.values()]
           finalrmclresult_adjusted_total_tmp =  [[(i, set().union(*[vals for _, vals in tmp]))] for i, tmp in enumerate(finalrmclresult)]
           finalrmclresult_adjusted_total = {k: sorted(list(v)) for sub in  finalrmclresult_adjusted_total_tmp for k, v in sub}
           log.info(f"Final result of rmcl after renumbering--{algorithm}: reverse branching-rmcl result with only the representative nodes: {finalrmclresult}, reverse branching-rmcl result with all the non core members: {finalrmclresult_adjusted_total}")
           return finalrmclresult_adjusted_total

def sr_mcl(dic_mclresult, originadj, defaultcorenum=0, coreinfoonly = True, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    mmoriginadj, clusmemlist, clussizelist, corecluscandlist = coreclusQ(dic_mclresult, originadj)    
    coreclushub, coreclusternumber, coreclus_values, coremapping, reverse_coremapping, _, _, _, _, _, _, _, _ = mclus_anaysis(mmoriginadj, clusmemlist, clussizelist, corecluscandlist, defaultcorenum = 0)
    corehubresult = {k: v for k, v in coreclus_values.items() if k[0] in [reverse_coremapping[coreclushub]] or k[1] in [reverse_coremapping[coreclushub]]}
    corehubresult_list = {tuple(int(x) for x in k): float(v) for k, v in corehubresult.items()}
    log.info(f"The adjacency across the members of the core cluster without the hub: {corehubresult_list}")
    corenonhubresult =  {k: v for k, v in coreclus_values.items() if not k[0] in [reverse_coremapping[coreclushub]] and not k[1] in [reverse_coremapping[coreclushub]]}
    corenonhubresult_list = {tuple(int(x) for x in k): float(v) for k, v in corenonhubresult.items()}
    log.info(f"The adjacency between the hub and the dangling nodes in the core cluster: {corenonhubresult_list}")
    combined = {**corenonhubresult, **corehubresult}
    coreclusreconnection = dict(sorted(combined.items()))
    renum_coreclusreconnection = {(reverse_coremapping[i], reverse_coremapping[j]): val for (i, j), val in coreclusreconnection.items()}
    renum_coreclusreconnection_list = {tuple(int(x) for x in k): float(v) for k, v in renum_coreclusreconnection.items()}
    log.info(f"The integrated adjacency sources in the corecluster are renumbered as an integer sequence for repeated MCL: {renum_coreclusreconnection_list}")
    max_row = max(r for r, c in renum_coreclusreconnection.keys()) + 1
    max_col = max(c for r, c in renum_coreclusreconnection.keys()) + 1
    shape = (max_row, max_col)
    rows, cols, data = zip(*[(r, c, v) for (r, c), v in renum_coreclusreconnection.items()])
    core_sparse_matrix = csr_matrix((data, (rows, cols)), shape=shape)
    srmcl_core_result = mclprocess(core_sparse_matrix)
    renum_srmcl_core_result = {k: [coremapping[x] for x in v] for k, v in srmcl_core_result.items()}
    log.info(f"The srmcl result focussing on the core cluter partionned: {renum_srmcl_core_result}")
    tmp_clusmemlist = clusmemlist.copy()
    tmp_clusmemlist.pop(coreclusternumber)
    dividedcore = [j for i,j in renum_srmcl_core_result.items()]
    srmcl_all_result_tmp= tmp_clusmemlist + dividedcore
    srmcl_all_result = sorted([sorted(group) for group in srmcl_all_result_tmp], key=lambda x: x[0] if x else float('inf'))
    log.info(f"The result of the srmcl is inserted into the non-core MCL cluster list and all clusters are newly sorted: {srmcl_all_result}")
    if coreinfoonly == True:
        return renum_srmcl_core_result
    else:
        return srmcl_all_result

def mixed_rmcl(dic_mclresult, originadj, threspruning, defaultcorenum = 0, branching = True, reverse_process = True, coreinfoonly = True, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    core_result = sr_mcl(dic_mclresult, originadj, defaultcorenum, coreinfoonly)
    log.info(f"The core cluster is the target of the SRMCL.")
    log.info(f"The core cluster core_result is: {core_result}")
    core_partition = sorted([sorted(group) for group in  [j for i,j in core_result.items()]], key=lambda x: x[0] if x else float('inf'))
    log.info(f"core cluster partitioning finished: {core_partition}")
    log.info(f"Now non core cluster partitioning started")
    if branching == True:
        log.info(f"Reverse branching MCL started.")
        non_core_result =  branching_rmcl(dic_mclresult, originadj, threspruning, defaultcorenum, reverse_process)
        log.info(f"non_core_result: {non_core_result}")
        non_core_partition = sorted([sorted(group) for group in  [j for i,j in non_core_result.items()]], key=lambda x: x[0] if x else float('inf'))
        log.info(f"non_core clusters partitioning finished: {non_core_partition}")
    else:
        log.info(f"Non core clusters insterted just as they are.")
        _, set_mcl_resultlist = mcldict_to_mclset(dic_mclresult)
        flattened_core_partition = sorted([item for sublist in core_partition for item in sublist])
        set_core_partition = {tuple(flattened_core_partition)}
        semi_non_core_result_tmp = set_mcl_resultlist-set_core_partition
        non_core_partition =  [sorted(tup) for tup in sorted(semi_non_core_result_tmp, key=lambda x: x[0])]
    all_result_tmp = core_partition + non_core_partition
    all_result = sorted([sorted(group) for group in all_result_tmp], key=lambda x: x[0] if x else float('inf'))
    log.info(f"partition merge: {all_result}")
    return all_result

def rmcl_basic(*args, **kwargs):
    return branching_rmcl(*args, **kwargs)

def find_all_in_dict_lists(
    d: Dict[Any, List[Any]], 
    target: Any
) -> List[Tuple[Any, List[Any]]]:
    result = []
    for key, lst in d.items():
        if target in lst:
            result.append((key, lst))
    return result

def clusinfo_from_nodes(clustering_result, node):
  if type(clustering_result) == dict:
    return find_all_in_dict_lists(clustering_result, node)
  elif  type(clustering_result) == tuple or type(clustering_result) == list:
    return find_all_in_dict_lists(dict(enumerate(clustering_result)),node)
  else:
    return []  

def append_hub_to_recluscore(rmclresultcore,hubnumlist):
    lenrmclresultcore = len(rmclresultcore)
    if type(hubnumlist) == int:
        tobeadded = [hubnumlist]
        lentobeadded = 1
    elif type(hubnumlist) == list:
        tobeadded = hubnumlist
        lentobeadded = len(tobeadded)    
    for i in range(lentobeadded):
        rmclresultcore[lenrmclresultcore + i] = [tobeadded[i]]
        print(i)
    return rmclresultcore

def log_communities_for_set_of_tuples(communities_set, label="detected tuples after sorting", logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    if not communities_set:
        log.info("%s: nothing", label)
        return
    sorted_comms = sorted(tuple(sorted(t)) for t in communities_set)
    lines = [f"{label}:"]
    for tup in sorted_comms:
        if len(tup) <= 10:  
            lines.append(f"  {tup}")
        else:
            chunks = [tup[i:i+8] for i in range(0, len(tup), 8)]
            first_line = f"  ({', '.join(map(str, chunks[0]))},"
            lines.append(first_line)
            for chunk in chunks[1:-1]:
                lines.append(f"   {', '.join(map(str, chunk))},") 
            last_chunk = chunks[-1]
            lines.append(f"   {', '.join(map(str, last_chunk))}),")   
    lines.append("  total: {} tuples".format(len(sorted_comms)))
    log.info("\n" + "\n".join(lines))

def mcldict_to_mclset(dic_mclresult_,logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    list_mclresult = sorted([sorted(group) for group in  [j for i,j in dic_mclresult_.items()]], key=lambda x: x[0] if x else float('inf'))
    log.info(f"The dict format for representing the MCL result was coverted into list type before the set calculation. list_mclresult: {list_mclresult}")
    set_mclresult = set(tuple(x) for x in list_mclresult)
    log.info(f"The set format treats lists as sets and is useful for operations that extract the intersection or union between two nested lists. set_mclresult: {set_mclresult}")
    log_communities_for_set_of_tuples(set_mclresult)
    return list_mclresult, set_mclresult

def mclset_to_mcldict(set_mclresult_,logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    list_mclresult = sorted(list([list(i) for i in set_mclresult_]))
    log.info(f"The set format for representing the MCL result was coverted into list. list_mclresult: {list_mclresult}")
    dict_mclresult = dict(enumerate(list_mclresult))
    log.info(f"The set format for representing the MCL result was coverted into dict. dict_mclresult: {dict_mclresult}")
    return list_mclresult, dict_mclresult

def mcldict_to_mcllist(dic_mclresult_, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    list_mclresult = sorted([sorted(group) for group in  [j for i,j in dic_mclresult_.items()]], key=lambda x: x[0] if x else float('inf'))
    log.info(f"The mcl result of which the type is dict was converted to a nested list mcllist: {list_mclresult}")
    return mcllist

def mcllist_to_mclset(list_mclresult_, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    set_mclresult = set(tuple(x) for x in list_mclresult_)
    log.info(f"The set format treats lists as sets and is useful for operations that extract the intersection or union between two nested lists. set_mclresult: {set_mclresult}")
    log_communities_for_set_of_tuples(set_mclresult)
    return set_mclresult

def mcllist_to_mcldict(list_mclresult_, logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    dict_mclresult = dict(enumerate(list_mclresult_))
    log.info(f"The list format for representing the MCL result was coverted into dict. dict_mclresult: {dict_mclresult}")
    return dict_mclresult

def mclset_to_mcllist(set_mclresult_,  logger = None):
    log = resolve_logger(logger, "mcl")
    print(f"log name: {log.name}")
    list_mclresult = sorted(list([list(i) for i in set_mclresult_]))
    log.info(f"The set format for representing the MCL result was coverted into list. list_mclresult: {list_mclresult}")
    return list_mclresult
   
