#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
import torch.nn.functional as F

from turtle import forward
from torch import Tensor, nn
from typing import List, Tuple
from torch import nn

CANO_NAME_MORANSI = 'moranI'
CANO_NAME_GEARYSC = 'gearyC'


class LinTranslator(nn.Module):
    """Linear translator with learnable softmax temperature.

    The weight matrix is always produced via column-wise softmax so that
    per-spot weights live on the probability simplex during optimisation.
    """

    def __init__(self,
                 dim_input: int,
                 dim_output: int,
                 tau: float=None,
                 seed: int=None,
                 device: torch.device=None
        ):
        super().__init__()
        self.seed = seed
        self.device = device
        if not seed is None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.trans = nn.Parameter(torch.randn(dim_input, dim_output))

    def _get_weight_mtx(self):
        tau = F.softplus(self.log_tau) + 0.01
        return torch.softmax(self.trans / tau, dim=0)

    def forward(self, input: Tensor, 
                      only_pred: bool=True):
        """
        Args:
            input (Tensor): Reference gene signature
            only_pred (bool, optional): not return weights or return. Defaults to True.

        Returns:
            : Translation results [translation weight matrix [Spot, Cell] ] 
        """
        trans = input @ self._get_weight_mtx()
        if only_pred:
            return trans
        
        return trans, self._get_weight_mtx().t()

class SpaStats(nn.Module):

    def __init__(self):
        super().__init__()
    
    def _standardize(self, X):
        return (X - torch.mean(X, dim=0, keepdim=True))  / torch.std(X, dim=0, keepdim=True)

class SparkX(SpaStats):

    def __init__(self,
                 locations: torch.Tensor,
        ):
        super().__init__()
        with torch.no_grad():
            self.locations = self._standardize(locations)
            self.inv_loc_gram = torch.inverse(self.locations.t() @ self.locations)    

    def sparkX(self, centered_expr):
        STS_inv = self.inv_loc_gram
        loc = self.locations
        N = centered_expr.shape[0]
        EHL = centered_expr.t() @ loc
        
        numerator = torch.sum((EHL @ STS_inv) * EHL, dim=1)
        denominator = torch.square(centered_expr).sum(axis=0)
        test_stats = numerator * N / denominator
        return test_stats

    def cal_spa_stats(self, gene_expr):
        gene_expr = self._standardize(gene_expr)
        return self.sparkX(gene_expr)
               

class SpaReg(SpaStats):
    
    def __init__(self, spa_adj, spa_lag, margin=0.01):
        super().__init__()
        self.spa_adj = spa_adj
        self.spa_lag = spa_lag
        self.criterian = nn.MarginRankingLoss(margin=margin)
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.mse = nn.MSELoss(reduce=None)

    def loss(self, Y_hat, score_fun='cosine'):
        spa_adj, Y_lag = self.spa_adj, self.spa_lag
        pred_lag = spa_adj @ Y_hat
        # return self.mse(pred_lag, Y_lag)
        neg_lag  = spa_adj @ Y_hat[np.random.permutation(Y_hat.shape[0])] # .detach()
        if score_fun == "bernoulli":
            pos_score = torch.sigmoid(torch.sum(Y_lag * pred_lag, dim=1))
            neg_score = torch.sigmoid(torch.sum(Y_lag * neg_lag, dim=1))
        if score_fun == "cosine":
            pos_score = self.cos_sim(Y_lag, pred_lag)
            neg_score = self.cos_sim(Y_lag, neg_lag)
        if score_fun == "euclidean":
            pos_score = torch.exp(- torch.sum( (Y_lag - pred_lag)**2, dim=1, keepdim=True))
            neg_score = torch.exp(- torch.sum( (Y_lag - neg_lag)**2, dim=1, keepdim=True))

        return self.criterian(pos_score,
                              neg_score, 
                              torch.ones_like(pos_score).sign())
            

class SpaAutoCorr(SpaStats):

    def __init__(self,
                 Y: Tensor,
                 spa_adj: torch.sparse_coo_tensor,
                 method: str=CANO_NAME_MORANSI,
        ):
        """Class for Spatial Regularization with Moran'S I

        Args:
            Y (Tensor): Training ST Matrix
            spa_adj (torch.sparse_coo_tensor): Spatial adjacency matrix
            method (str, optional): Spatial Autocorrelation method. Defaults to CANO_NAME_MORANSI.

        Raises:
            Exception: _description_
        """
        super().__init__()
        self.spa_adj = spa_adj
        self.adjmat_degrees = torch.sparse.sum(spa_adj, dim=1).to_dense()
        if not method in {CANO_NAME_MORANSI, CANO_NAME_GEARYSC}:
            raise Exception("Unimplemented autocorrelation method!")
        self.method = method
        with torch.no_grad():
            self.truth_stats = self.cal_spa_stats(Y)
        self.mse = nn.MSELoss()

    def _moransI_numerator_sparse_adj(self, centered_expr, sparse_adj):
        numerator = torch.sum(centered_expr * torch.sparse.mm(sparse_adj, centered_expr.t()).t(), dim=1)
        return numerator

    def _gearysI_numerator_sparse_adj(self, 
                                      gene_expr: Tensor, 
                                      sparse_adj: torch.sparse_coo_tensor):
        """_summary_
        
        I use a trick from Laplacian Eigenmaps for computing the numerator
        $\sum_{i,j} w_ij ||z_i - z_j||^2 = 2 * Trace(Z(D - W)Z^T)$
        To test:
        
        ```
        from scipy import sparse
        _W = np.abs(np.random.randn(4, 4))
        W = _W @ _W.T
        W -= np.diag(W.diagonal())
        Y = np.random.randn(4, 1)
        D = W.sum(axis=1)
        np.sum((Y - Y.T)**2 * W), 2 * np.sum(D * Y.flatten() ** 2) - 2 * Y.T @ W @ Y
        ```
        
        Args:
            gene_expr (Tensor): Expression matrix
            sparse_adj (SparseTensor): Sparse spatial adjacency matrix

        Returns:
            Tensor: numerator of spatial autocorrelation value
        """
        
        degrees = self.adjmat_degrees
        # Trace(ZDZ^T), since z_i is 1-d vector, we can simplify the computation as follows
        traceZDZ_T = (degrees.view(1,-1) * torch.square(gene_expr)).sum(dim=1)
        # Trace(ZWZ^T), since z_i is 1-d vector, we can simplify the computation as follows
        traceZWZ_T = torch.sum(gene_expr * torch.sparse.mm(sparse_adj, gene_expr.t()).t(), dim=1)
        numerator = 2 * (traceZDZ_T - traceZWZ_T)
        return numerator    

    def cal_spa_stats(self, gene_expr: Tensor):
        """Calculate spatial autocorrelation statistics

        Args:
            gene_expr (Tensor): Expression matrix

        Returns:
            Tensor: Spatial autocorrelation statistics
        """
        method = self.method
        gene_expr = gene_expr.t()
        sparse_adj = self.spa_adj
        W = torch.sum(sparse_adj._values())
        N = gene_expr.shape[1]
        centered_expr = gene_expr - torch.mean(gene_expr, dim=1, keepdim=True)
        denominator = torch.sum(torch.square(centered_expr), dim=1)
        
        if (denominator == 0).any():
            mask = torch.zeros_like(denominator)
            mask[denominator == 0] = 1e-8
            denominator = denominator + mask

        if method == CANO_NAME_MORANSI:
            numerator_I = self._moransI_numerator_sparse_adj(centered_expr, sparse_adj)
            return N/W * numerator_I / denominator

        if method == CANO_NAME_GEARYSC:
            numerator_C = self._gearysI_numerator_sparse_adj(gene_expr, sparse_adj)
            return (N - 1)/ (2*W) * numerator_C / denominator
        return None    

    def loss(self, Y_hat: Tensor):
        """Calculate spatial regularization loss

        Args:
            Y_hat (Tensor): predicted ST expression matrix

        Returns:
            Scalar: mean squared error loss of spatial autocorrelation statistics
        """
        pred_stats = self.cal_spa_stats(Y_hat)
        return self.mse(pred_stats, self.truth_stats)

    def loss_batch(self, Y_hat_batch: Tensor, spot_indices: Tensor):
        """Calculate spatial regularization loss for a batch of spots.

        Uses the subgraph of the adjacency matrix induced by the batch spots
        to compute approximate autocorrelation statistics, compared against
        the pre-computed ground truth stats for those spots.

        Args:
            Y_hat_batch (Tensor): predicted expression for batch spots [batch_size, Gene]
            spot_indices (Tensor): indices of spots in this batch

        Returns:
            Scalar: MSE loss of batch spatial autocorrelation statistics
        """
        idx = spot_indices
        sub_adj = torch.index_select(self.spa_adj, 0, idx)
        sub_adj = torch.index_select(sub_adj, 1, idx)

        gene_expr = Y_hat_batch.t()
        W = torch.sum(sub_adj._values())
        if W == 0:
            return torch.tensor(0.0, device=Y_hat_batch.device)
        N = gene_expr.shape[1]
        centered_expr = gene_expr - torch.mean(gene_expr, dim=1, keepdim=True)
        denominator = torch.sum(torch.square(centered_expr), dim=1)

        mask = torch.zeros_like(denominator)
        mask[denominator == 0] = 1e-8
        denominator = denominator + mask

        if self.method == CANO_NAME_MORANSI:
            numerator = torch.sum(centered_expr * torch.sparse.mm(sub_adj, centered_expr.t()).t(), dim=1)
            pred_stats = N / W * numerator / denominator
        elif self.method == CANO_NAME_GEARYSC:
            degrees = torch.sparse.sum(sub_adj, dim=1).to_dense()
            traceZDZ_T = (degrees.view(1, -1) * torch.square(gene_expr)).sum(dim=1)
            traceZWZ_T = torch.sum(gene_expr * torch.sparse.mm(sub_adj, gene_expr.t()).t(), dim=1)
            numerator = 2 * (traceZDZ_T - traceZWZ_T)
            pred_stats = (N - 1) / (2 * W) * numerator / denominator
        else:
            return torch.tensor(0.0, device=Y_hat_batch.device)

        truth_batch = self.truth_stats[spot_indices] if self.truth_stats.dim() == 1 else self.truth_stats
        return self.mse(pred_stats, truth_batch)


class TransDeconv(nn.Module):
    """Translation-based cell type deconvolution     
    """

    def __init__(self,
                 dim_tgt_outputs: int,
                 dim_ref_inputs:  int,
                 n_feats: int,
                 tau: float=0.5,
                 spa_autocorr: SpaAutoCorr=None,
                 scaler_g_init: Tensor=None,
                 scaler_s_init: Tensor=None,
                 raw_counts: bool=True,
                 device:    torch.device=None,
                 seed:       int=None
                ):
        """
        Args:
            dim_tgt_outputs (int): Dimension of ground truth gene profile
            dim_ref_inputs (int): Dimension of reference gene signature
            n_feats (int): number of genes
            spa_autocorr (SpaAutoCorr, optional): Instance of Spatial index class. Default to None.
            scaler_g_init (Tensor, optional): Fixed per-gene platform-effect log-scaler [1, n_feats].
            scaler_s_init (Tensor, optional): Fixed per-spot library-size log-scaler [dim_tgt_outputs, 1].
            raw_counts (bool): Whether the target data is raw counts (selects loss type).
            device (torch.device, optional): Device of computation. Defaults to None.
            seed (int, optional): Seed number. Defaults to None.
        """
        super().__init__()
        self.device = device
        self.seed = seed
        self.raw_counts = raw_counts
        if not seed is None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        if scaler_g_init is not None:
            self.register_buffer('scaler_g', scaler_g_init)
        else:
            self.register_buffer('scaler_g', torch.zeros(1, n_feats))
        if scaler_s_init is not None:
            self.register_buffer('scaler_s', scaler_s_init)
        else:
            self.register_buffer('scaler_s', torch.zeros(dim_tgt_outputs, 1))

        self.bias_g = nn.Parameter(torch.zeros(1, n_feats))
        self.cos_by_col = nn.CosineSimilarity(dim=1)
        self.cos_by_row = nn.CosineSimilarity(dim=0)
        self.mse = nn.MSELoss(reduction='mean')
        self.trans = LinTranslator(dim_ref_inputs, 
                                        dim_tgt_outputs,
                                        tau=tau,
                                        device=device,
                                        seed=seed)
        self.spa_autocorr = spa_autocorr
        
   
    def predict(self, X: Tensor, 
                      return_cluster: bool=False,
                ) -> np.ndarray:
        """Do translation

        Args:
            X (Tensor): Reference gene signature
            return_cluster (bool, optional): Whether or not to return tranlation matrix. Defaults to False.

        Returns:
            np.ndarray: Translation result
        """
        self.eval()
        with torch.no_grad():
            preds, weight = self(X)
            
        if return_cluster:
            return preds.cpu().numpy(), weight.cpu().numpy()
        return preds.cpu().numpy()

    def loss(self, 
             X: Tensor, 
             Y: Tensor, 
             cls_abd_sig: Tensor,
             truth_autocorr: Tensor=None,
             wt_l2_G: float=2.0,
             wt_l2_S: float=2.0,
             wt_l1:   float=5.0,
             wt_abd:  float=2.0,
             wt_spa:  float=1.0,
             wt_recon: float=0.01,
             method_autocorr: str=CANO_NAME_MORANSI):
        """Calculate translation loss given ground truths

        Args:
            X (Tensor): Reference gene signature matrix [Cell, Gene]
            Y (Tensor): Ground truth gene profile matrix [Spot, gene]
            cls_abd_sig (Tensor): Class abundance signature.
            truth_autocorr (Tensor): Ground-Truth spatal autocorrelation indices. Defaults to None.
            wt_l2_G (float): Weight of l2 regularization. Defaults to 2.0.
            wt_l2_S (float): Weight of l2 regularization. Defaults to 2.0.
            wt_l1 (float): weight of l1 regulariztion. Defaults to 5.0.
            wt_abd (float): weight of abundance signature loss. Defaults to 2.0.
            wt_spa (float): weight spatial regularization loss. Defaults to 1.0.
            wt_recon (float): weight of reconstruction loss. Defaults to 1.0.
            method_autocorr (str): name of spatial autocorrelation methods "moranI" or "gearyC"

        Returns:
            Tensor: scalar loss
        """
        Y_hat = torch.exp(self.scaler_g) * self.trans(X.t()).t() * torch.exp(self.scaler_s)
        
        norm_Y_hat = torch.norm(Y_hat, dim=1)
        sel_valid = (norm_Y_hat != 0) & ~torch.isnan(norm_Y_hat) &  ~torch.isinf(norm_Y_hat)

        loss1 = (1 - self.cos_by_col(Y_hat[sel_valid], Y[sel_valid])).mean() 
        loss2 = (1 - self.cos_by_row(Y_hat[sel_valid], Y[sel_valid])).mean()
        loss3 = (1 - self.cos_by_col(torch.sum(Y_hat[sel_valid], dim=0, keepdim=True), torch.sum(Y[sel_valid], dim=0, keepdim=True))).mean()
        loss4 = (1 - self.cos_by_row(torch.sum(Y_hat[sel_valid], dim=1, keepdim=True), torch.sum(Y[sel_valid], dim=1, keepdim=True))).mean()
        imp_loss = loss1  + loss2 + loss3 + loss4

        if self.raw_counts:
            Y_hat_clamped = torch.clamp(Y_hat[sel_valid], min=1e-8)
            recon_loss = F.poisson_nll_loss(Y_hat_clamped, Y[sel_valid], log_input=False)
        else:
            recon_loss = F.mse_loss(torch.log1p(torch.clamp(Y_hat[sel_valid], min=0)),
                                    torch.log1p(torch.clamp(Y[sel_valid], min=0)))

        W = self.trans._get_weight_mtx()
        l1norm = torch.norm(W, p=1, dim=1).mean()

        abd_sig_hat = torch.sum(W, dim=1, keepdim=True)
        abd_sig = cls_abd_sig
        abd_norm = (1 - self.cos_by_row(torch.log2(abd_sig_hat + 1)/ np.log2(2), 
                                      torch.log2(abd_sig + 1)  / np.log2(2))).mean()
        if (not self.spa_autocorr is None) and (not truth_autocorr is None):
            pred_auto_cor = self.spa_autocorr.cal_spa_stats(Y_hat, method_autocorr)
            spa_reg = self.mse(truth_autocorr, pred_auto_cor)
        else:
            spa_reg = 0
        
        loss = imp_loss + wt_recon * recon_loss + \
                wt_l1 * l1norm + wt_abd * abd_norm + wt_spa * spa_reg
        return loss
        
        
    def forward(self, X: Tensor):
        """Translate

        Args:
            X (Tensor): Reference gene signature [Cell, Gene]

        Returns:
            Tuple[Tensor]: Prediction and translation weight matrix
        """
        # X is ? by gene
        preds, weight = self.trans(X.t(), False)
        Y_hat = torch.exp(self.scaler_g) * preds.t() * torch.exp(self.scaler_s)
        return Y_hat, weight


class RaTranslator(nn.Module):
    """Linear translator
    """

    def __init__(self,
                 dim_input: float,
                 dim_output: float,
                 dim_hid: float=512,
                 seed: int=None,
                 device: torch.device=None
        ):
        """

        Args:
            dim_input (float): dimension of reference gene profile
            dim_output (float): dimension of target gene profile
            dim_hid  (int): dimension of hidden layer. Defaults to 512.
            seed (int, optional): random seed. Defaults to None.
            device (torch.device, optional): device of computation. Defaults to None.
        """

        super().__init__()
        self.seed = seed
        self.device = device
        if not seed is None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        # self.trans1 = nn.Parameter(torch.randn(dim_input, dim_hid))
        # self.relu = nn.ReLU()
        # self.trans2 = nn.Parameter(torch.randn(dim_hid, dim_output))
        # self.scaler = nn.Parameter(torch.ones(1, dim_output))
        self.trans = nn.Parameter(torch.randn(dim_input, dim_output))

    def _get_weight_mtx(self):
        return torch.softmax(self.trans, dim=0)
        # return torch.sigmoid(self.trans1) @ torch.sigmoid(self.trans2, dim=0)

    def sparse_reg(self):
        return 0

    def scale_reg(self):
        return 0

    def forward(self, input: Tensor, 
                      only_pred: bool=True):
        """

        Args:
            input (Tensor): Reference gene signature
            only_pred (bool, optional): not return weights or return. Defaults to True.

        Returns:
            : Translation results [translation weight matrix [Spot, Cell] ] 
        """
        trans = input @ self._get_weight_mtx()
        # trans = torch.square(self.scaler) * self.relu(self.relu(input @ self.trans1) @ self.trans2)
        if only_pred:
            return trans
        
        return trans, self._get_weight_mtx().t()      

    def forward_batch(self, input: Tensor, spot_indices: Tensor):
        """Forward pass for a subset of spots.

        Args:
            input (Tensor): Reference gene signature [Gene, Cell]
            spot_indices (Tensor): Indices of spots in this batch

        Returns:
            Tensor: Translation results for selected spots [Gene, batch_size]
        """
        W_batch = torch.softmax(self.trans[:, spot_indices], dim=0)
        return input @ W_batch


class RaTranslatorLowRank(nn.Module):
    """Linear translator
    """

    def __init__(self,
                 dim_input: int,
                 dim_output: int,
                 dim_hid: int=512,
                 clip_max: float=10,
                 seed: int=None,
                 device: torch.device=None
        ):
        """

        Args:
            dim_input (int): dimension of reference gene profile
            dim_output (int): dimension of target gene profile
            dim_hid  (int): dimension of hidden layer. Defaults to 512.
            clip_max (float): clipping threshold for output of first translation. Defaults to 10.
            seed (int, optional): random seed. Defaults to None.
            device (torch.device, optional): device of computation. Defaults to None.
        """

        super().__init__()
        self.seed = seed
        self.device = device
        self.clip_max = clip_max
        if not seed is None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.trans1 = nn.Parameter(torch.randn(dim_input, dim_hid))
        self.trans2 = nn.Parameter(torch.randn(dim_hid, dim_output))

    def _get_weight_mtx(self):
        return torch.clip(torch.square(self.trans1), max=self.clip_max) @ torch.softmax(self.trans2, dim=0)

    def transform(self, input):
        return input @ torch.clip(torch.square(self.trans1), max=self.clip_max) @ torch.softmax(self.trans2, dim=0)

    def sparse_reg(self):
        return torch.norm(torch.square(self.trans1), p=1, dim=0).mean() 
        
    def scale_reg(self):
        return torch.norm(torch.square(self.trans1), p=2, dim=0).mean()

    def forward(self, input: Tensor, 
                      only_pred: bool=True):
        """

        Args:
            input (Tensor): Reference gene signature
            only_pred (bool, optional): not return weights or return. Defaults to True.

        Returns:
            : Translation results [translation weight matrix [Spot, Cell] ] 
        """
        trans = self.transform(input)
        # trans = torch.square(self.scaler) * self.relu(self.relu(input @ self.trans1) @ self.trans2)
        if only_pred:
            return trans
        
        return trans, self._get_weight_mtx().t()   

    def forward_batch(self, input: Tensor, spot_indices: Tensor):
        """Forward pass for a subset of spots.

        Args:
            input (Tensor): Reference gene signature [Gene, Cell]
            spot_indices (Tensor): Indices of spots in this batch

        Returns:
            Tensor: Translation results for selected spots [Gene, batch_size]
        """
        W1 = torch.clip(torch.square(self.trans1), max=self.clip_max)
        W2_batch = torch.softmax(self.trans2[:, spot_indices], dim=0)
        return input @ W1 @ W2_batch

class RaTranslatorNoneLinear(nn.Module):
    """Linear translator
    """

    def __init__(self,
                 dim_input: float,
                 dim_output: float,
                 dim_hid: float=512,
                 seed: int=None,
                 device: torch.device=None
        ):
        """

        Args:
            dim_input (float): dimension of reference gene profile
            dim_output (float): dimension of target gene profile
            dim_hid  (int): dimension of hidden layer. Defaults to 521.
            seed (int, optional): random seed. Defaults to None.
            device (torch.device, optional): device of computation. Defaults to None.
        """

        super().__init__()
        self.seed = seed
        self.device = device
        
        if not seed is None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.trans1 = nn.Parameter(torch.randn(dim_input, dim_hid))
        self.trans2 = nn.Parameter(torch.randn(dim_hid, dim_output))
        self.relu = nn.ReLU()

    def transform(self, input):
        return self.relu(self.relu(input @ self.trans1) @ self.trans2)

    def sparse_reg(self):
        return torch.norm(torch.square(self.trans1), p=1, dim=0).mean() 
        
    def scale_reg(self):
        return torch.norm(torch.square(self.trans1), p=2, dim=0).mean()

    def forward(self, input: Tensor, 
                      only_pred: bool=True):
        """

        Args:
            input (Tensor): Reference gene signature
            only_pred (bool, optional): not return weights or return. Defaults to True.

        Returns:
            : Translation results [translation weight matrix [Spot, Cell] ] 
        """
        trans = self.transform(input)
        # trans = torch.square(self.scaler) * self.relu(self.relu(input @ self.trans1) @ self.trans2)
        return trans

    def forward_batch(self, input: Tensor, spot_indices: Tensor):
        """Forward pass for a subset of spots.

        Args:
            input (Tensor): Reference gene signature [Gene, Cell]
            spot_indices (Tensor): Indices of spots in this batch

        Returns:
            Tensor: Translation results for selected spots [Gene, batch_size]
        """
        return self.relu(self.relu(input @ self.trans1) @ self.trans2[:, spot_indices])


class TransImp(nn.Module):
    """Translation-based spatial gene imputation     
    """

    def __init__(self,
                 dim_tgt_outputs: int,
                 dim_ref_inputs:  int,
                 dim_hid: int=512,
                 clip_max: float=10,
                 spa_inst: SpaAutoCorr=None,
                 mapping_mode: str='full',
                 device:    torch.device=None,
                 seed:       int=None
                ):
        """

        Args:
            dim_tgt_outputs (int): Dimension of ground truth gene profile
            dim_ref_inputs (int): Dimension of reference gene signature
            dim_hid  (int): dimension of hidden layer. Defaults to 512.
            clip_max (float): clipping threshold for output of first translation. Defaults to 10.
            spa_inst (SpaAutoCorr, optional): Instance of Spatial index class. Default to None.
            mapping_mode (str): "lowrank" or "full". Defaults to full.
            device (torch.device, optional): Device of computation. Defaults to None.
            seed (int, optional): Seed number. Defaults to None.
        """
        super().__init__()
        self.device = device
        self.seed = seed
        if not seed is None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        
        self.cos_by_col = nn.CosineSimilarity(dim=1)
        self.cos_by_row = nn.CosineSimilarity(dim=0)
        self.mse = nn.MSELoss(reduction='mean')
        if mapping_mode == 'full':
            self.trans = RaTranslator(dim_ref_inputs, 
                                    dim_tgt_outputs,
                                    device=device,
                                    seed=seed)
        elif mapping_mode == "lowrank":
            self.trans = RaTranslatorLowRank(
                                    dim_ref_inputs, 
                                    dim_tgt_outputs,
                                    clip_max=clip_max,
                                    dim_hid=dim_hid,
                                    device=device,
                                    seed=seed)
        elif mapping_mode == "nonelinear":
            self.trans = RaTranslatorNoneLinear(
                dim_ref_inputs,
                dim_tgt_outputs,
                dim_hid=dim_hid,
                device=device,
                seed=seed
            )
        else:
            raise Exception(f'Unimplemented mapping mode {mapping_mode}')
        self.spa_inst = spa_inst
   
    def predict(self, X: Tensor) -> np.ndarray:
        """Do translation

        Args:
            X (Tensor): Reference gene signature
            return_cluster (bool, optional): Whether or not to return tranlation matrix. Defaults to False.

        Returns:
            np.ndarray: Translation result
        """
        self.eval()
        with torch.no_grad():
            preds = self(X)
            preds = preds / max(torch.std(preds).item(), 1e-6)
        return preds.cpu().numpy()

    def predict_raw(self, X: Tensor) -> Tensor:
        """Translate without per-call std normalization.

        Use this when multiple predictions (e.g. spliced and unspliced)
        must share the same scale so their relative magnitudes are preserved.

        Args:
            X (Tensor): Reference gene signature [Cell, Gene]

        Returns:
            Tensor: Raw translation result (stays on device)
        """
        self.eval()
        with torch.no_grad():
            return self(X)

    def predict_batched(self, X: Tensor, batch_size: int=8192) -> np.ndarray:
        """Chunked prediction for large numbers of spots.

        Args:
            X (Tensor): Reference gene signature [Cell, Gene]
            batch_size (int): Number of spots per chunk. Defaults to 8192.

        Returns:
            np.ndarray: Translation result [Spot, Gene]
        """
        self.eval()
        n_spots = self.trans.trans.shape[1] if hasattr(self.trans, 'trans') else self.trans.trans2.shape[1]
        chunks = []
        with torch.no_grad():
            for start in range(0, n_spots, batch_size):
                end = min(start + batch_size, n_spots)
                idx = torch.arange(start, end, device=X.device)
                chunk = self.trans.forward_batch(X.t(), idx).t()
                chunks.append(chunk)
            preds = torch.cat(chunks, dim=0)
            preds = preds / max(torch.std(preds).item(), 1e-6)
        return preds.cpu().numpy()

    def _calibration_scale(self, train_X: Tensor, train_Y: Tensor) -> float:
        """Compute a global scale factor from training genes where ground truth exists."""
        raw_train = self(train_X)
        return train_Y.std().item() / max(raw_train.std().item(), 1e-6)

    def predict_calibrated(self, test_X: Tensor,
                           train_X: Tensor, train_Y: Tensor) -> np.ndarray:
        """Translate and calibrate output scale to match observed ST data.

        Computes a global scale factor from training genes (where ground truth
        is available) and applies it to test gene predictions, so the imputed
        expression lives on the same scale as the observed ST.

        Args:
            test_X (Tensor): Reference signature for test genes [Cell/Cluster, test_Gene]
            train_X (Tensor): Reference signature for training genes [Cell/Cluster, train_Gene]
            train_Y (Tensor): Observed ST expression for training genes [Spot, train_Gene]

        Returns:
            np.ndarray: Calibrated predictions [Spot, test_Gene]
        """
        self.eval()
        with torch.no_grad():
            scale = self._calibration_scale(train_X, train_Y)
            preds = self(test_X) * scale
        return preds.cpu().numpy()

    def predict_batched_calibrated(self, test_X: Tensor,
                                   train_X: Tensor, train_Y: Tensor,
                                   batch_size: int=8192) -> np.ndarray:
        """Chunked calibrated prediction for large numbers of spots.

        Same as predict_calibrated but processes spots in chunks to avoid OOM.

        Args:
            test_X (Tensor): Reference signature for test genes [Cell/Cluster, test_Gene]
            train_X (Tensor): Reference signature for training genes [Cell/Cluster, train_Gene]
            train_Y (Tensor): Observed ST expression for training genes [Spot, train_Gene]
            batch_size (int): Number of spots per chunk. Defaults to 8192.

        Returns:
            np.ndarray: Calibrated predictions [Spot, test_Gene]
        """
        self.eval()
        n_spots = self.trans.trans.shape[1] if hasattr(self.trans, 'trans') else self.trans.trans2.shape[1]
        with torch.no_grad():
            scale = self._calibration_scale(train_X, train_Y)
            chunks = []
            for start in range(0, n_spots, batch_size):
                end = min(start + batch_size, n_spots)
                idx = torch.arange(start, end, device=test_X.device)
                chunk = self.trans.forward_batch(test_X.t(), idx).t()
                chunks.append(chunk)
            preds = torch.cat(chunks, dim=0) * scale
        return preds.cpu().numpy()

    def loss(self, 
             X: Tensor, 
             Y: Tensor, 
             wt_l1norm: float=1e-2,
             wt_l2norm: float=1e-2,
             wt_spa:  float=1e-1,
             gene_weights=1,
             truth_spa_stats: Tensor=None,
             ):
        """Calculate translation loss given ground truths

        Args:
            X (Tensor): Reference gene signature matrix [Cell, Gene]
            Y (Tensor): Ground truth gene profile matrix [Spot, gene]
            wt_l1norm (float): l1 normalization for translation function. Default to 1e-2. 
            wt_l2norm (float): l2 normalization for translation function. Default to 1e-2.
            wt_spa (float): weight for spatial regularization loss. Default to 1e-1.
            truth_spa_stats (Tensor): Ground-Truth spatial statistics. Defaults to None.

        Returns:
            Tensor: scalar loss
        """
        Y_hat = self.trans(X.t()).t()
        
        norm_Y_hat = torch.norm(Y_hat, dim=1)
        sel_valid = (norm_Y_hat != 0) & ~torch.isnan(norm_Y_hat) &  ~torch.isinf(norm_Y_hat)

        loss1 = (1 - self.cos_by_col(Y_hat[sel_valid], Y[sel_valid])).mean() 
        loss2 = ((1 - self.cos_by_row(Y_hat[sel_valid], Y[sel_valid])) * gene_weights).mean()        
        imp_loss = loss1 + loss2 
        
        if not wt_l1norm is None and wt_l1norm > 0:
            imp_loss = imp_loss + wt_l1norm * self.trans.sparse_reg()
        if not wt_l2norm is None and wt_l2norm > 0:
            imp_loss = imp_loss + wt_l2norm * self.trans.scale_reg()

        loss = imp_loss
        
        if not self.spa_inst is None and wt_spa > 0:
            spa_reg = self.spa_inst.loss(Y_hat)
            loss = loss + wt_spa * spa_reg           
        
        return loss, imp_loss.item(), spa_reg.item() if not self.spa_inst is None and wt_spa > 0 else 0

    def loss_batch(self,
                   X: Tensor,
                   Y_batch: Tensor,
                   spot_indices: Tensor,
                   wt_l1norm: float=1e-2,
                   wt_l2norm: float=1e-2,
                   wt_spa: float=1e-1,
                   gene_weights=1,
                   ):
        """Calculate translation loss on a mini-batch of spots.

        Uses forward_batch to only compute predictions for the selected spots,
        enabling training on datasets with millions of spots.

        Args:
            X (Tensor): Reference gene signature matrix [Cell, Gene]
            Y_batch (Tensor): Ground truth for batch spots [batch_size, Gene]
            spot_indices (Tensor): Indices of spots in this batch
            wt_l1norm (float): L1 regularization weight. Default to 1e-2.
            wt_l2norm (float): L2 regularization weight. Default to 1e-2.
            wt_spa (float): Spatial regularization weight. Default to 1e-1.
            gene_weights: Per-gene weights for column cosine loss. Default to 1.

        Returns:
            Tuple: (loss, imp_loss_value, spa_reg_value)
        """
        Y_hat_batch = self.trans.forward_batch(X.t(), spot_indices).t()

        norm_Y_hat = torch.norm(Y_hat_batch, dim=1)
        sel_valid = (norm_Y_hat != 0) & ~torch.isnan(norm_Y_hat) & ~torch.isinf(norm_Y_hat)

        loss1 = (1 - self.cos_by_col(Y_hat_batch[sel_valid], Y_batch[sel_valid])).mean()
        loss2 = ((1 - self.cos_by_row(Y_hat_batch[sel_valid], Y_batch[sel_valid])) * gene_weights).mean()
        imp_loss = loss1 + loss2

        if not wt_l1norm is None and wt_l1norm > 0:
            imp_loss = imp_loss + wt_l1norm * self.trans.sparse_reg()
        if not wt_l2norm is None and wt_l2norm > 0:
            imp_loss = imp_loss + wt_l2norm * self.trans.scale_reg()

        loss = imp_loss
        spa_reg = 0

        if not self.spa_inst is None and wt_spa > 0:
            spa_reg = self.spa_inst.loss_batch(Y_hat_batch, spot_indices)
            loss = loss + wt_spa * spa_reg

        spa_val = spa_reg.item() if isinstance(spa_reg, Tensor) else spa_reg
        return loss, imp_loss.item(), spa_val
        
        
    def forward(self, X: Tensor):
        """Translate

        Args:
            X (Tensor): Reference gene signature [Cell, Gene]

        Returns:
            Tuple[Tensor]: Prediction and translation weight matrix
        """
        # X is ? by gene
        preds = self.trans(X.t())
        Y_hat = preds.t()
        return Y_hat

    def forward_batch(self, X: Tensor, spot_indices: Tensor):
        """Translate for a subset of spots.

        Args:
            X (Tensor): Reference gene signature [Cell, Gene]
            spot_indices (Tensor): Indices of target spots

        Returns:
            Tensor: Predicted expression for batch spots [batch_size, Gene]
        """
        preds = self.trans.forward_batch(X.t(), spot_indices)
        return preds.t()