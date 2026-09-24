#author: alex sun
#date: 12/20/2021
#based on 
#https://github.com/CUAI/CorrectAndSmooth/blob/b910314a59270984f5e249462ee3faa815fc9a0c/outcome_correlation.py#L77

import torch
import torch.nn.functional as F
import os

import numpy as np
import random

import scipy.sparse as sp
from scipy.sparse import linalg
from tqdm import tqdm
from torch import matmul
import collections
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OrderedSet(collections.abc.Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)

def process_adj(data):
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index

    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    return adj, deg_inv_sqrt

def gen_normalized_adjs(adj, D_isqrt):
    DAD = D_isqrt.view(-1,1)*adj*D_isqrt.view(1,-1)
    DA = D_isqrt.view(-1,1) * D_isqrt.view(-1,1)*adj
    AD = adj*D_isqrt.view(1,-1) * D_isqrt.view(1,-1)
    return DAD, DA, AD

def gen_normalized_adj(adj, pw): # pw = 0 is D^-1A, pw=1 is AD^-1
    deg = adj.sum(dim=1).to(torch.float)
    front  = deg.pow(-(1-pw))
    front[front == float('inf')] = 0
    back  = deg.pow(-(pw))
    back[back == float('inf')] = 0
    return (front.view(-1,1)*adj*back.view(1,-1))

def get_Gamma(adj,omega):
    """
    S = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2    
    D = diag(A 1)
    @param adj, adj matrix with no self-loop
    @param omega, smoothing factor
    @return:
    #eq. (5) In Jia and Benson, Jia, Junteng, and Austin R. Benson. 
    # "A unifying generative model for graph learning algorithms: Label propagation, 
    # graph convolutions, and combinations." arXiv preprint arXiv:2101.07730 (2021).
    # 
    # beta * torch.eye(S.size(0)).to(device) - beta * alpha * S
    #06042021, change to Eq. 2.33 in Jia and Benson 2020
    #02282022, revised based on definition of N in Jia&Bension p.6, line 1
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1)) #get node degree
    d_inv_sqrt = np.power(d, -0.5).flatten() 
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) #conver to diagonal
    #S -> d_mat_inv_sqrt*adj*d_mat_inv_sqrt, I-D^(1/2) W D(-1/2), here S=N 
    S = sp.eye(adj.shape[0])-adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    #as 0301, change to use G=I+omega*N
    #G = beta * (torch.eye(S.size(0)).to(device) - alpha * S)    
    G = sp.eye(adj.shape[0]) + omega*S
    G = torch.tensor(G.toarray(),dtype=torch.double).to(device)
    return G

def getS(adj):
    """
    # L = D^-1/2 (A) D^-1/2 
    # D = diag(A 1)
    :param adj:
    :return:
    S, Symmetrically normalized adjacency matrix
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1)) #turn to dense array?
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    #D^-1/2 (A) D^-1/2
    S = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    S = torch.tensor(S.toarray(),dtype=torch.double).to(device)
    return S

def labelupdate(idx_train, idx_test, pred_train, Gamma):    
    """
    This implements Eq. (2.33) in Jia and Benson, 2021, Gamma=I+omega*N
    """
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)    
    #(I+wN)_{U x U}
    test_val_Gamma = Gamma[idx_test, :][:, idx_test] #Gamma_UU
    #Gamma_{U x L} * y_{L x nT}
    prodUL = matmul(Gamma[idx_test, :][:, idx_train], pred_train.t()) #Gamma_UL*y_L    
    #Inverse U x U
    dd = torch.inverse(test_val_Gamma) #inv(Gamma_UU)   
    #U x nT
    updated_pred_test = -matmul(dd, prodUL) #-inv(Gamma_UU)*Gamma_UL*y_L, Eq. (2.14)

    return updated_pred_test.t() #nT by U



def gp_interpolate(idx_train, idx_test, res_pred_train):
    idx_L = torch.LongTensor(idx_train) #labeled sites
    idx_U = torch.LongTensor(idx_test)   #unlabeled sites
    

def lp_refine(idx_unlabeled, idx_obs, rL, output, adj, omega, **kwargs):
    """
    Params
    ------
    idx_obs, indices of observed nodes
    idx_unlabeled, indices of unobserved nodes
    output, model output
    adj, adjacency mat
    alpha, beta, parameter controlling smoothing level
    #02282022, revise based on Jia&Benson2021
    """
    Gamma = get_Gamma(adj,omega) #this is I+omega*N
    
    pred_test = output[:, idx_unlabeled] #y_U
    #calculate mismatch at observed nodes    
    #propagate the residual
    kwargs.setdefault("isNormalize", False)
    if kwargs["isNormalize"]:
        yU  = labelupdate(idx_obs, idx_unlabeled, rL, Gamma)*kwargs['std']+kwargs['mean']
        """
        #not used
        qt = kwargs['transformer']
        #convert to numpy and then back to tensor?
        refined_test = pred_test + torch.from_numpy(qt.inverse_transform(interpolate(idx_obs, idx_unlabeled, rL, Gamma).data.cpu().numpy())).to(torch.double).to(device)
        """
    else:
        yU = labelupdate(idx_obs, idx_unlabeled, rL, Gamma)
    #03032022, only update if value is greater than 0
    refined_test = pred_test+yU
    refined_test[refined_test<0] = pred_test[refined_test<0]

    return refined_test

def diffuse(x, adj, num_propagations, p, alpha):
    S = getS(adj)
    #residual error
    if p is None:
        p = 1.
    if alpha is None:
        alpha = 0.5

    x = x **p

    x = torch.from_numpy(x).to(torch.double).to(device)
    
    for i in tqdm(range(num_propagations)):
#       x = (1-args.alpha)* inital_features + args.alpha * adj @ x
        x = x - alpha * matmul(x, torch.eye(S.shape[0]).to(device) - S)
        x = x **p
    return x

def multifidelity(idx_unlabeled, idx_obs, Y_H, Y_L,adj,omega):
    Gamma = get_Gamma(adj,omega) #this is pytorch tensor
    #Han and Gortz, AIAA 2012
    #low fidelity forecast at sampled sites
    F = Y_L[:,idx_obs]
    R_L = Gamma[idx_obs,:][:,idx_obs]    
    invR_L = torch.inverse(R_L)    
    beta0 = torch.zeros(F.shape[0],1).double().to(device)
    for i in range(F.shape[0]):
        fvec = F[i,:]        
        c = matmul(matmul(fvec,invR_L),fvec.t())
        beta0[i] = matmul(matmul(fvec,invR_L),(Y_H[i,:].t()))/c
    print (beta0)
    Y_E = torch.zeros((Y_L.shape[0],len(idx_unlabeled))).double().to(device)
    residual =  Y_H-beta0.repeat(1,len(idx_obs))*F 
    for ix,i in enumerate(idx_unlabeled):
        mu = beta0*(Y_L[:,i].unsqueeze(1))
        p1 = matmul(Gamma[idx_obs,i].t(),invR_L).unsqueeze(0)
        Y_E[:,ix] = (mu+matmul(p1,residual.t()).t()).squeeze()

    return Y_E

def OI(idx_unlabeled, idx_obs, Y_H, Y_L,adj,omega,tau=1e-3):
    H = get_Gamma(adj,omega) #this is pytorch tensor
    PH = H[:, idx_obs]
    R_L = H[idx_obs,:][:,idx_obs] + tau* torch.eye(len(idx_obs)).to(device)
    invR_L = torch.inverse(R_L)    
    residual = Y_H-Y_L[:,idx_obs]

    Y_E = Y_L+matmul(matmul(PH,invR_L),residual.t()).t()
    return Y_E
