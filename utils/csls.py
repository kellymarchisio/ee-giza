# By Kelly Marchisio

import numpy as np
from . import matops
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances_chunked


def get_avg_dists(X, Y, metric='cosine', knn=10):
    def _reduce_func(D_chunk, start):
        divisor = min(D_chunk.shape[1], knn)
        D_chunk = matops.keep_bottomk(D_chunk, knn)
        return D_chunk.sum(axis=1) / float(divisor)
    result = pairwise_distances_chunked(X, Y, metric=metric,
            reduce_func=_reduce_func, n_jobs=20)
    all_results = [i for i in result]
    return np.concatenate(all_results)


def calculate_csls_scores(S2T, T2S, csls_knn=10, topk=10):
    '''
        Calculate CSLS Scores.

        Args:
            S2T: Source-to-target matrix that is the result of S @ T.T
            T2S: Target-to-source matrix that is the result of T @ S.T

        Test Function: 
	X = array([[0.5488135 , 0.71518937, 0.60276338],
       [0.54488318, 0.4236548 , 0.64589411],
       [0.43758721, 0.891773  , 0.96366276],
       [0.38344152, 0.79172504, 0.52889492],
       [0.56804456, 0.92559664, 0.07103606]])

	Y = array([[0.4359949 , 0.02592623, 0.54966248],
       [0.43532239, 0.4203678 , 0.33033482],
       [0.20464863, 0.61927097, 0.29965467],
       [0.26682728, 0.62113383, 0.52914209],
       [0.13457995, 0.51357812, 0.18443987],
       [0.78533515, 0.85397529, 0.49423684],
       [0.84656149, 0.07964548, 0.50524609]])

	correct answer = 
	array([[-0.06194635,  0.12796299,  0.06188674,  0.10739143,  0.01510211,
         0.11357157, -0.06945745],
       [ 0.22720448,  0.09905055, -0.13841618,  0.02025804, -0.235944  ,
         0.03269671,  0.16495328],
       [-0.04454164,  0.03872445,  0.07918305,  0.17502332,  0.02310511,
         0.00107526, -0.21902755],
       [-0.25620973,  0.0836979 ,  0.1709998 ,  0.16241586,  0.15496893,
         0.09376397, -0.29415056],
       [-0.66372336,  0.04335364,  0.12342963, -0.07856674,  0.18231056,
         0.14002129, -0.38016711]])	

    '''
    # If topk > 0, reduce the elements output to the sparse graph to
    # the topk nearest neighbors.
    S2Tavgsims = 1 - get_avg_dists(S2T, T2S, metric='cosine', knn=csls_knn)
    T2Savgsims = 1 - get_avg_dists(T2S, S2T, metric='cosine', knn=csls_knn)
    def _reduce_func(D_chunk, start):
        S2Tavgsims_slice = S2Tavgsims.T[start:start+D_chunk.shape[0]]
        S2Tavgsims_slice = np.expand_dims(S2Tavgsims_slice, 1)
        # Just subtracting S2Tavgsims_slice does same thing as subtracting the
        # expanded form (np.repeat(S2Tavgsims_slice, D_chunk.shape[1], axis=1))
        result_chunk = 2*(1-D_chunk) - S2Tavgsims_slice - T2Savgsims
        if topk:
            result_chunk = matops.keep_topk(result_chunk, topk) 
        return sparse.csr_matrix(result_chunk, dtype=np.float32)
    result = pairwise_distances_chunked(S2T, T2S, metric='cosine',
            reduce_func=_reduce_func, n_jobs=20, working_memory=2048)
    return result
