#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import random
import numpy as np

from turtle import forward
from torch import Tensor, nn
from typing import List, Tuple
from torch import nn

CANO_NAME_MORANSI = 'moranI'
CANO_NAME_GEARYSC = 'gearyC'

class NNTranslator(nn.Module):
    """Neural Translator
    Translate reference gene signature to 
    gene profile at the spot level.
    """

    def __init__(self,
                 dim_input: int,
                 dim_hid: int,
                 dim_output: int,
                 seed: int=None,
                 device: torch.device=None
        ):
        """

        Args:
            dim_input (int): Dimension of reference gene signature
            dim_hid (int):   Dimension of bottleneck hidden layer
            dim_output (int): Dimension of target gene profile
            seed (int, optional):  Random seed number. Defaults to None.
            device (torch.device, optional): Device of computation. Defaults to None.
        """
        super().__init__()
        self.seed = seed
        self.device = device
        if not seed is None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.regressor = nn.Sequential(
            nn.Linear(dim_input, dim_hid),
            nn.GELU(),
            nn.Linear(dim_hid, dim_output),
            nn.ReLU()
        )

    def forward(self, input):
        return self.regressor(input)


class TransImpute(nn.Module):
    """Translation-based missing gene imputation
    """

    def __init__(self,
                 dim_tgt_outputs: int,
                 dim_ref_inputs:  int,
                 dim_hid_AE:  int=512,
                 dim_hid_RG:  int=128,
                 device:    torch.device=None,
                 seed:       int=None
                ):
        """
        Args:
            dim_tgt_outputs (int): Dimension of target gene profile
            dim_ref_inputs (int): Dimension of reference gene signature
            dim_hid_AE (int, optional): Dimension of bottelneck hidden layer of VAE. Defaults to 512.
            dim_hid_RG (int, optional):  Dimension of bottleneck hidden layer NNTrans. Defaults to 128.
            device (torch.device, optional):  Random seed number. Defaults to None.
            seed (int, optional): Device of computation. Defaults to None.
        """
                
        super().__init__()
        self.device = device
        self.seed = seed
        if not seed is None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.encoder = nn.Sequential(
            nn.Linear(dim_ref_inputs, dim_hid_AE),
            nn.GELU(),
            nn.Linear(dim_hid_AE, dim_hid_AE),
            nn.GELU(),
        )

        self.fc_mu = nn.Linear(dim_hid_AE, dim_hid_AE)
        self.fc_var = nn.Linear(dim_hid_AE, dim_hid_AE)

        nn.init.uniform_(self.fc_var.weight, -0.08, 0.08)

        self.decoder = nn.Sequential(
            nn.Linear(dim_hid_AE, dim_ref_inputs),
        )
        
        self.cos_by_col = nn.CosineSimilarity(dim=1)
        self.cos_by_row = nn.CosineSimilarity(dim=0)
        self.mse = nn.MSELoss(reduction='mean')
        self.nn_reg  = NNTranslator(dim_hid_AE, dim_hid_RG, dim_tgt_outputs, seed, device)
   
    def encode(self, X: Tensor) -> Tuple[Tensor]:
        """encode input

        Args:
            X (Tensor): Gene By Cell matrix

        Returns:
            Tuple[Tensor]: Latent mu and log var.
        """
        Z = self.encoder(X)
        mu = self.fc_mu(Z)
        logvar = self.fc_var(Z)
        return mu, logvar

    def decode(self, Z: Tensor) -> Tensor:
        """Decode latent Z

        Args:
            Z (Tensor): Encodings of reference gene profile

        Returns:
            Tensor: Reconstructed reference gene signature
        """
        return self.decoder(Z)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (Tensor): Mean of the latent Gaussian [Cell x latent dim]
            logvar (Tensor): Standard deviation of the latent Gaussian [Cell x latent dim]

        Returns:
            Tensor: Samples after reparameterization
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    
    def predict(self,
                test_X: Tensor, 
                ) -> np.ndarray:
        """Predict given reference gene signatures

        Args:
            test_X (Tensor): Gene signatures

        Returns:
            ndarray: Prediction results
        """
        self.eval()
        with torch.no_grad():
            preds = self(test_X).cpu().numpy()
        return preds

    def _vae_loss(self, Z: Tensor, 
                        X: Tensor, 
                        mu: Tensor, 
                        logvar: Tensor) -> tuple:
        """Compute VAE loss for model generalization

        Args:
            Z (Tensor): Latent encodings
            X (Tensor): Input reference signature
            mu (Tensor): Mean parameter of Q(Z|input)
            logvar (Tensor): LogVar parameter of Q(Z|input)

        Returns:
            tuple(scaler, scaler): AE loss and KLD loss
        """
        X_hat = self.decode(Z)
        ae_loss = self.mse(X_hat.t(), X)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return ae_loss, kld_loss

    def _nn_trans_loss(self, Z: Tensor, 
                             Y: Tensor):
        """Translation loss

        Args:
            Z (Tensor): Encodings of reference gene signature
            Y (Tensor): Ground Truth spatial gene profile

        Returns:
           Scaler : Translation loss
        """
        Y_hat = self.nn_reg(Z).t()       

        norm_Y_hat = torch.norm(Y_hat, dim=1)
        sel_valid = (norm_Y_hat != 0) & ~torch.isnan(norm_Y_hat) &  ~torch.isinf(norm_Y_hat)

        loss1 = (1 - self.cos_by_col(Y_hat[sel_valid], Y[sel_valid])).mean() 
        loss2 = (1 - self.cos_by_row(Y_hat[sel_valid], Y[sel_valid])).mean()
        loss3 = (1 - self.cos_by_col(torch.sum(Y_hat[sel_valid], dim=0, keepdim=True), torch.sum(Y[sel_valid], dim=0, keepdim=True))).mean()
        loss4 = (1 - self.cos_by_row(torch.sum(Y_hat[sel_valid], dim=1, keepdim=True), torch.sum(Y[sel_valid], dim=1, keepdim=True))).mean()
        loss = loss1  + loss2 + loss3 + loss4
        return loss
        
    
    def _get_vae_trans_loss(self, 
                         X: Tensor, 
                         Y: Tensor, 
                         reg_dims: List[int],
                         ) -> Tuple[float]:
        """Get losses

        Args:
            X (Tensor): Reference gene signature matrix
            Y (Tensor): Ground truth gene profile matrix
            reg_dims (List[int]): Dimensions of supervised training

        Returns:
            Tuple[float]: [AE loss, KLD loss, Trans loss]
        """
        mu, logvar = self.encode(X.t())
        torch.clamp_(logvar, max=50)
        Z = self.reparameterize(mu, logvar)
        ae_loss, kld_loss = self._vae_loss(Z, X, mu, logvar)
      
        trans_loss = self._nn_trans_loss(Z[reg_dims], Y)
        return ae_loss, kld_loss, trans_loss
    
    def loss_vae_trans(self, 
                       X: Tensor, 
                       Y: Tensor, 
                       reg_dims: List[int],
                       ae_weight: float=1.0,
                       kld_weight: float=1.0
        ) -> Tuple[float]:
        """Calculate training losses

        Args:
            X (Tensor): Reference gene signature matrix
            Y (Tensor): Ground truth gene profile matrix
            reg_dims (List[int]): Dimensions of supervised training
            ae_weight (float, optional): Weight for AE loss. Defaults to 1.0.
            kld_weight (float, optional): Weight for KLD loss. Defaults to 1.0.

        Returns:
            Tuple[float]: total loss, translation loss, AE loss, KLD loss
        """
        ae_loss, kld_loss, trans_loss = self._get_vae_trans_loss(X, Y, reg_dims)
        tt_loss = ae_weight * (ae_loss) + kld_weight * kld_loss + trans_loss
        return tt_loss, trans_loss, ae_loss, kld_loss
    
        
    def forward(self, X: Tensor) -> Tensor:
        """Translate gene signature

        Args:
            X (Tensor): Reference gene signature [Cell, Gene]

        Returns:
            Tensor: Translated gene profile
        """
        mu, _ = self.encode(X.t())        
        return self.nn_reg(mu).t()  


class LinTranslator(nn.Module):
    """Linear translator
    """

    def __init__(self,
                 dim_input: float,
                 dim_output: float,
                 seed: int=None,
                 device: torch.device=None
        ):
        """

        Args:
            dim_input (float): dimension of reference gene profile
            dim_output (float): dimension of target gene profile
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

        self.trans = nn.Parameter(torch.randn(dim_input, dim_output))

    def _get_weight_mtx(self):
        return torch.square(self.trans)

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
                 Y,
                 spa_adj: torch.sparse_coo_tensor,
                 method: str=CANO_NAME_MORANSI,
        ):
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

    def _gearysI_numerator_sparse_adj(self, gene_expr, sparse_adj):
        # I use a trick from Laplacian Eigenmaps for computing the numerator
        # \sum_{i,j} w_ij ||z_i - z_j||^2 = 2 * Trace(Z(D - W)Z^T)
        # To test:
        # from scipy import sparse
        # _W = np.abs(np.random.randn(4, 4))
        # W = _W @ _W.T
        # W -= np.diag(W.diagonal())
        # Y = np.random.randn(4, 1)
        # D = W.sum(axis=1)
        # np.sum((Y - Y.T)**2 * W), 2 * np.sum(D * Y.flatten() ** 2) - 2 * Y.T @ W @ Y
        degrees = self.adjmat_degrees
        # Trace(ZDZ^T), since z_i is 1-d vector, we can simplify the computation as follows
        traceZDZ_T = (degrees.view(1,-1) * torch.square(gene_expr)).sum(dim=1)
        # Trace(ZWZ^T), since z_i is 1-d vector, we can simplify the computation as follows
        traceZWZ_T = torch.sum(gene_expr * torch.sparse.mm(sparse_adj, gene_expr.t()).t(), dim=1)
        numerator = 2 * (traceZDZ_T - traceZWZ_T)
        return numerator    

    def cal_spa_stats(self, gene_expr):
        method = self.method
        gene_expr = gene_expr.t()
        sparse_adj = self.spa_adj
        W = torch.sum(sparse_adj._values())
        N = gene_expr.shape[1]
        centered_expr = gene_expr - torch.mean(gene_expr, dim=1, keepdim=True)
        denominator = torch.sum(torch.square(centered_expr), dim=1)
        
        if (denominator == 0).any():
            mask = torch.zeros_like(denominator)
            mask[denominator == 0] = 1e-6
            denominator = denominator + mask

        if method == CANO_NAME_MORANSI:
            numerator_I = self._moransI_numerator_sparse_adj(centered_expr, sparse_adj)
            return N/W * numerator_I / denominator

        if method == CANO_NAME_GEARYSC:
            numerator_C = self._gearysI_numerator_sparse_adj(gene_expr, sparse_adj)
            return (N - 1)/ (2*W) * numerator_C / denominator
        return None    

    def loss(self, Y_hat):
        pred_stats = self.cal_spa_stats(Y_hat)
        # with no_grad
        # self.cal_spa_stats(Y_hat[np.random.permutation(Y_hat.shape[0])].detach())
        return self.mse(pred_stats, self.truth_stats)


class LocSpaAutoCorr(SpaStats):

    def __init__(self,
                 Y,
                 spa_adj: torch.sparse_coo_tensor,
                 truth_stats: torch.Tensor=None,
                 method: str=CANO_NAME_MORANSI,
        ):
        super().__init__()
        self.spa_adj = spa_adj
        self.truth_stats = truth_stats
        self.adjmat_degrees = torch.sparse.sum(spa_adj, dim=1).to_dense()
        if not method in {CANO_NAME_MORANSI, CANO_NAME_GEARYSC}:
            raise Exception("Unimplemented autocorrelation method!")
        self.method = method
        # self.mse = nn.MSELoss()
        self.mse = nn.SmoothL1Loss()
        with torch.no_grad():
            self.truth_stats = self.cal_spa_stats(Y)
        
    def _moransI_numerator_sparse_adj(self, centered_expr, sparse_adj):
        numerator = centered_expr * torch.sparse.mm(sparse_adj, centered_expr.t()).t()
        return numerator

    def _gearysI_numerator_sparse_adj(self, gene_expr, sparse_adj):
        # I use a trick from Laplacian Eigenmaps for computing the numerator
        # \sum_{i,j} w_ij ||z_i - z_j||^2 = 2 * Trace(Z(D - W)Z^T)
        # To test:
        # from scipy import sparse
        # _W = np.abs(np.random.randn(4, 4))
        # W = _W @ _W.T
        # W -= np.diag(W.diagonal())
        # Y = np.random.randn(4, 1)
        # D = W.sum(axis=1)
        # np.sum((Y - Y.T)**2 * W), 2 * np.sum(D * Y.flatten() ** 2) - 2 * Y.T @ W @ Y
        degrees = self.adjmat_degrees
        # Trace(ZDZ^T), since z_i is 1-d vector, we can simplify the computation as follows
        traceZDZ_T = (degrees.view(1,-1) * torch.square(gene_expr)).sum(dim=1)
        # Trace(ZWZ^T), since z_i is 1-d vector, we can simplify the computation as follows
        traceZWZ_T = torch.sum(gene_expr * torch.sparse.mm(sparse_adj, gene_expr.t()).t(), dim=1)
        numerator = 2 * (traceZDZ_T - traceZWZ_T)
        return numerator    

    def cal_spa_stats(self, gene_expr):
        method = self.method
        gene_expr = gene_expr.t()
        sparse_adj = self.spa_adj
        W = torch.sum(sparse_adj._values())
        N = gene_expr.shape[1]
        centered_expr = gene_expr - torch.mean(gene_expr, dim=1, keepdim=True)
        denominator = torch.sum(torch.square(centered_expr), dim=1)
        
        if (denominator == 0).any():
            mask = torch.zeros_like(denominator)
            mask[denominator == 0] = 1e-6
            denominator = denominator + mask

        if method == CANO_NAME_MORANSI:
            numerator_I = self._moransI_numerator_sparse_adj(centered_expr, sparse_adj)
            # print(numerator_I.shape, denominator.shape)
            return N * numerator_I / denominator.view(-1, 1)

        if method == CANO_NAME_GEARYSC:
            numerator_C = self._gearysI_numerator_sparse_adj(gene_expr, sparse_adj)
            return (N - 1)/ (2*W) * numerator_C / denominator
        return None  

    def loss(self, Y_hat):
        pred_stats = self.cal_spa_stats(Y_hat)
        # with no_grad
        # self.cal_spa_stats(Y_hat[np.random.permutation(Y_hat.shape[0])].detach())
        return self.mse(pred_stats, self.truth_stats)

    

class TransDeconv(nn.Module):
    """Translation-based cell type deconvolution     
    """

    def __init__(self,
                 dim_tgt_outputs: int,
                 dim_ref_inputs:  int,
                 n_feats: int,
                 spa_autocorr: SpaAutoCorr=None,
                 device:    torch.device=None,
                 seed:       int=None
                ):
        """

        Args:
            dim_tgt_outputs (int): Dimension of ground truth gene profile
            dim_ref_inputs (int): Dimension of reference gene signature
            n_feats (int): number of genes
            spa_lap (torch.sparse_coo_tensor, optional): The laplacian matrix for smoothing. Defaults to None.
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
        
        self.scaler_g = nn.Parameter(torch.zeros(1, n_feats))
        self.bias_g = nn.Parameter(torch.zeros(1, n_feats))
        self.scaler_s = nn.Parameter(torch.zeros(dim_tgt_outputs, 1))
        self.cos_by_col = nn.CosineSimilarity(dim=1)
        self.cos_by_row = nn.CosineSimilarity(dim=0)
        self.mse = nn.MSELoss(reduction='mean')
        self.trans = LinTranslator(dim_ref_inputs, 
                                        dim_tgt_outputs,
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
             method_autocorr: str=CANO_NAME_MORANSI):
        """Calculate translation loss given ground truths

        Args:
            X (Tensor): Reference gene signature matrix [Cell, Gene]
            Y (Tensor): Ground truth gene profile matrix [Spot, gene]
            cls_abd_sig (Tensor): Class abundance signateure

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

        abd_sig_hat = torch.sum(self.trans.trans ** 2, dim=1, keepdim=True)
        abd_sig =  cls_abd_sig
        
        W = torch.square(self.trans.trans.t())
        l1norm = torch.norm(W, p=1, dim=1).mean()
        l2normG = torch.norm(torch.exp(self.scaler_g), p=2)
        l2normS = torch.norm(torch.exp(self.scaler_s), p=2)
        abd_norm = (1 - self.cos_by_row(torch.log2(abd_sig_hat + 1)/ np.log2(2), 
                                      torch.log2(abd_sig + 1)  / np.log2(2))).mean()
        if (not self.spa_autocorr is None) and (not truth_autocorr is None):
            pred_auto_cor = self.spa_autocorr.cal_spa_stats(Y_hat, method_autocorr)
            spa_reg = self.mse(truth_autocorr, pred_auto_cor)
        else:
            spa_reg = 0
        
        loss = imp_loss +  wt_l2_G * l2normG + wt_l2_S * l2normS + \
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
        # Y_hat = (self.scaler_g * preds.t() + self.bias_g) * self.scaler_s
        # Y_hat = (torch.exp(self.scaler_g) * preds.t() + self.bias_g ** 2) * torch.exp(self.scaler_s)
        Y_hat = torch.exp(self.scaler_g) * preds.t() * torch.exp(self.scaler_s)
        # Y_hat = (torch.exp(self.scaler_g) * self.nn_reg(X.t()).t() + self.bias_g**2) * torch.exp(self.scaler_s)
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


class RaTranslatorLowRank(nn.Module):
    """Linear translator
    """

    def __init__(self,
                 dim_input: float,
                 dim_output: float,
                 dim_hid: float=512,
                 clip_max: float=10,
                 seed: int=None,
                 device: torch.device=None
        ):
        """

        Args:
            dim_input (float): dimension of reference gene profile
            dim_output (float): dimension of target gene profile
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


class TransImp(nn.Module):
    """Translation-based spatial gene imputation     
    """

    def __init__(self,
                 dim_tgt_outputs: int,
                 dim_ref_inputs:  int,
                 dim_hid: int=512,
                 clip_max: float=10,
                 spa_inst=None,
                 mapping_mode: str='full',
                 device:    torch.device=None,
                 seed:       int=None
                ):
        """

        Args:
            dim_tgt_outputs (int): Dimension of ground truth gene profile
            dim_ref_inputs (int): Dimension of reference gene signature
            n_feats (int): number of genes
            spa_lap (torch.sparse_coo_tensor, optional): The laplacian matrix for smoothing. Defaults to None.
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

    def loss(self, 
             X: Tensor, 
             Y: Tensor, 
            #  cls_abd_sig: Tensor,
            #  wt_l2_G: float=2.0,
            #  wt_l2_S: float=2.0,
            #  wt_l1:   float=5.0,
            #  wt_abd:  float=2.0,
             wt_l1norm: float=1e-2,
             wt_l2norm: float=1e-2,
             wt_spa:  float=1e-1,
             truth_spa_stats: Tensor=None,
             ):
        """Calculate translation loss given ground truths

        Args:
            X (Tensor): Reference gene signature matrix [Cell, Gene]
            Y (Tensor): Ground truth gene profile matrix [Spot, gene]
            cls_abd_sig (Tensor): Class abundance signateure

        Returns:
            Tensor: scalar loss
        """
        # Y_hat = torch.exp(self.scaler_g) * self.trans(X.t()).t() * torch.exp(self.scaler_s)
        Y_hat = self.trans(X.t()).t()
        
        norm_Y_hat = torch.norm(Y_hat, dim=1)
        sel_valid = (norm_Y_hat != 0) & ~torch.isnan(norm_Y_hat) &  ~torch.isinf(norm_Y_hat)

        loss1 = (1 - self.cos_by_col(Y_hat[sel_valid], Y[sel_valid])).mean() 
        loss2 = (1 - self.cos_by_row(Y_hat[sel_valid], Y[sel_valid])).mean()

        
        # loss3 = (1 - self.cos_by_col(torch.sum(Y_hat[sel_valid], dim=0, keepdim=True), torch.sum(Y[sel_valid], dim=0, keepdim=True))).mean()
        # loss4 = (1 - self.cos_by_row(torch.sum(Y_hat[sel_valid], dim=1, keepdim=True), torch.sum(Y[sel_valid], dim=1, keepdim=True))).mean()
        # imp_loss = loss1  + loss2 + loss3 + loss4
        imp_loss = loss1 + loss2 

        # imp_loss = self.mse(Y_hat, Y)
        if not wt_l1norm is None and wt_l1norm > 0:
            imp_loss = imp_loss + wt_l1norm * self.trans.sparse_reg()
        if not wt_l2norm is None and wt_l2norm > 0:
            imp_loss = imp_loss + wt_l2norm * self.trans.scale_reg()

        # abd_sig_hat = torch.sum(self.trans.trans ** 2, dim=1, keepdim=True)
        # abd_sig =  cls_abd_sig
        
        
        # l1norm = torch.norm(W, p=1, dim=1).mean()
        # l2normG = torch.norm(torch.exp(self.scaler_g), p=2)
        # l2normS = torch.norm(torch.exp(self.scaler_s), p=2)
        # abd_norm = (1 - self.cos_by_row(torch.log2(abd_sig_hat + 1)/ np.log2(2), 
        #                               torch.log2(abd_sig + 1)  / np.log2(2))).mean()

        loss = imp_loss
        # if not self.spa_inst is None and not truth_spa_stats is None and wt_spa:
        #     pred_spa_stats = self.spa_inst.cal_spa_stats(Y_hat)
        #     spa_reg = self.mse(truth_spa_stats, pred_spa_stats)
        #     loss = loss + wt_spa * spa_reg    
        if not self.spa_inst is None and wt_spa > 0:
            spa_reg = self.spa_inst.loss(Y_hat)
            loss = loss + wt_spa * spa_reg

            
        
        # loss = imp_loss +  wt_l2_G * l2normG + wt_l2_S * l2normS + \
        #         wt_l1 * l1norm + wt_abd * abd_norm + wt_spa * spa_reg
        
        
        return loss, imp_loss.item(), spa_reg.item() if not self.spa_inst is None and wt_spa > 0 else 0
        
        
    def forward(self, X: Tensor):
        """Translate

        Args:
            X (Tensor): Reference gene signature [Cell, Gene]

        Returns:
            Tuple[Tensor]: Prediction and translation weight matrix
        """
        # X is ? by gene
        preds = self.trans(X.t())
        # Y_hat = (self.scaler_g * preds.t() + self.bias_g) * self.scaler_s
        # Y_hat = (torch.exp(self.scaler_g) * preds.t() + self.bias_g ** 2) * torch.exp(self.scaler_s)
        Y_hat = preds.t()
        # Y_hat = (torch.exp(self.scaler_g) * self.nn_reg(X.t()).t() + self.bias_g**2) * torch.exp(self.scaler_s)
        return Y_hat