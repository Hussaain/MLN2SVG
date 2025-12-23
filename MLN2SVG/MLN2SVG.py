from utils import refine_spatial_domains
import numpy as np
import torch
import random
import os
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from model import GCAE
from torch.optim import Adam
import torch.nn as nn
from utils import apply_mclust_R, reconstruct_spatial_expression, refine_spatial_domains
import matplotlib.pyplot as plt
from tqdm import trange
from anndata import AnnData
from typing import Optional
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse, csr_matrix
from utils import custom_palette


class MLN2SVG:
    def __init__(self,
                 adata,
                 num_cluster,
                 alpha=0.5,
                 input_dim=14,  # 15
                 z_dim=14,
                 output_dim=14,
                 MLN2SVG_seed=100,
                 device='cuda',
                 lr=1e-4,
                 n_epochs=40000,
                 copy: bool = True,
                 refine: bool = True,
                 ):
        super(MLN2SVG, self).__init__()

        # Set seed
        random.seed(MLN2SVG_seed)
        torch.manual_seed(MLN2SVG_seed)
        np.random.seed(MLN2SVG_seed)
        self.device = device

        self.gcae = GCAE(input_dim, z_dim, output_dim).to(self.device)
        self.lr = lr
        self.epochs = n_epochs
        self.loss_func = nn.MSELoss()
        self.optimizer = Adam(self.gcae.parameters(), lr=self.lr)

        self.adata = adata.copy() if copy else adata
        self.num_cluster = num_cluster
        self.alpha = alpha
        self.indim = input_dim
        self.zdim = z_dim
        self.outdim = output_dim
        self.refine = refine
        
    def load_data(self):
        """Load and preprocess spatial data including reconstruction and graph building"""
        print("Stage 1: Spatial data reconstruction in progress...")
    
    # Perform spatial reconstruction with updated variable names
        adata_processed, exp_matrix_original, exp_matrix_augmented, smooth_transition_graph = reconstruct_spatial_expression(adata_original=self.adata,
                                                                                                                             alpha=self.alpha,
                                                                                                                             n_components=self.zdim )
                                                                                                
                                                                                                     
        print("Spatial graph assembled—ready for the next step!")
    
        return adata_processed, exp_matrix_original, exp_matrix_augmented,smooth_transition_graph
    
    def train_MLN2SVG(self, exp_matrix_original, exp_matrix_augmented, smooth_transition_graph):
        """Train the MLN2SVG model with original and augmented expression data"""
        print("Phase 2: MLN2SVG Models training has benn started...")
        
        # Convert data to tensors and move to device
        smooth_transition_graph = torch.Tensor(smooth_transition_graph).to(self.device)
        original_data = torch.Tensor(exp_matrix_original).to(self.device)
        augmented_data = torch.Tensor(exp_matrix_augmented).to(self.device)
        
        # Initialize training progress bar
        pbar = trange(self.epochs)
        
        for epoch in pbar:
            # Forward pass
            latent_original, latent_augmented, reconst_original, reconst_augmented = self.gcae(
                original_data, augmented_data, smooth_transition_graph
            )
            
            # Calculate composite loss
            contrastive_loss = 0.5 * (self.loss_func(latent_original, latent_augmented) + 
                                    self.loss_func(reconst_original, reconst_augmented))
            recon_loss = 1.5 * (self.loss_func(reconst_original, original_data) + 
                              self.loss_func(reconst_augmented, augmented_data))
            total_loss = (contrastive_loss + recon_loss).to(self.device)
            
            # Update progress description
            pbar.set_description(f'Training Loss: {total_loss.item():.4f}')
            
            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # Final inference pass
        with torch.no_grad():
            latent_original, latent_augmented, reconst_original, reconst_augmented = self.gcae(
                original_data, augmented_data, smooth_transition_graph
            )
        
        print("✓ Model training completed successfully")
        return latent_original, latent_augmented, reconst_original, reconst_augmented 

    # def compute_clusters(self, processed_adata, latent_original):
    #     """Perform clustering on the latent representations"""
    #     print("Stage 3: Clustering initiated...")
        
    #     # Prepare expression matrix
    #     expression_df = pd.DataFrame(
    #         processed_adata.X.toarray()[:, ],
    #         index=processed_adata.obs.index,
    #         columns=processed_adata.var.index
    #     )
        
    #     # Prepare latent representations
    #     cells = np.array(expression_df.index)
    #     latent_reps = pd.DataFrame(latent_original.cpu().numpy())
    #     latent_reps.index = cells
        
    #     # Store latent representations in adata
    #     processed_adata.obsm['MLN2SVG'] = latent_reps.loc[processed_adata.obs_names,].values
        
    #     # Perform clustering
    #     sc.pp.neighbors(
    #         processed_adata,
    #         n_neighbors=10,
    #         use_rep='MLN2SVG'
    #     )
    #     processed_adata = apply_mclust_R(
    #         processed_adata,
    #         num_cluster=self.num_cluster
    #     )
        
    #     # Optional spatial refinement
    #     if self.refine:
    #         processed_adata.obs['mln2svg_clust'] = refine_spatial_domains(processed_adata.obs['mclust'], coord=processed_adata.obsm['spatial'], n_neighbors=10 ) 
    #     else:
    #         pass
    #     return processed_adata 
    
    def compute_clusters(self, processed_adata, latent_original):
        """Perform clustering on the latent representations"""
        print("Stage 3: Clustering initiated...")
        
        # Create expression matrix DataFrame
        expression_matrix = pd.DataFrame(
            processed_adata.X.toarray(),
            index=processed_adata.obs.index,
            columns=processed_adata.var.index
        )
        
        # Prepare latent representations DataFrame
        cells = np.array(expression_matrix.index)
        latent_representations = pd.DataFrame(latent_original.cpu().numpy())
        latent_representations.index = cells
        
        # Store latent representations in adata object
        processed_adata.obsm['MLN2SVG'] = latent_representations.loc[processed_adata.obs_names].values
        
        # Perform nearest neighbors and clustering
        sc.pp.neighbors(
            processed_adata,
            n_neighbors=10,
            use_rep='MLN2SVG'
        )
        processed_adata = apply_mclust_R(
            processed_adata,
            num_cluster=self.num_cluster
        )
        
        # Optional spatial refinement
        if self.refine:
            processed_adata.obs['mln2svg_clust'] = refine_spatial_domains(
                processed_adata.obs['mclust'],
                coord=processed_adata.obsm['spatial'],
                n_neighbors=10
            )
        
        print("✓ Clustering completed successfully")
        return processed_adata   
        


    def plot_clusters(self, processed_adata):
        fig, axs = plt.subplots(figsize=(8, 8))
        sc.pl.spatial(
            processed_adata,
            img_key='hires',
            color='mln2svg_clust',
            size=1.3,
            alpha=0.9,
            title='MLN2SVG Spatial Domains',
            palette=custom_palette,            # <- make sure this is defined globally
            legend_loc='right margin',
            show=False,
            ax=axs
        )
    
        plt.savefig('spatial_domains.png', dpi=600, bbox_inches='tight')
        