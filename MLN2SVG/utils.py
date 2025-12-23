import os
import logging
import numpy as np
import pandas as pd
from typing import Optional
from contextlib import contextmanager
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse, csr_matrix
import scanpy as sc
from natsort import natsorted
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri, numpy2ri
#from rpy2.robjects.conversion import localconverter, default_converter
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import matplotlib.pyplot as plt
import scanpy as sc
import numba
from scipy.sparse import issparse
from collections import Counter
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from wordcloud import WordCloud


def reconstruct_spatial_expression(
        adata_original: AnnData,
        alpha: float = 1,
        n_neighbors: int = 10,
        n_pcs: int = 15,
        use_highly_variable: Optional[bool] = True,
        normalize_total: bool = True,  
        copy: bool = True,
        n_components: int = 20,
):
    """Reconstruct spatial data using graph-based smoothing"""
    
    # Initialize data
    adata = adata_original.copy() if copy else adata_original
    adata.layers['counts'] = adata.X  

    # Normalization
    if normalize_total:
        sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.layers['log1p_original'] = adata.X

    # Feature selection and PCA
    hvg_genes = adata.var['highly_variable']
    hvg_genes = list(hvg_genes[hvg_genes.values].index)
    
    exp_matrix_original = adata.to_df(layer='log1p_original')[hvg_genes].to_numpy()
    pca_original = PCA(n_components=n_components)
    pca_original.fit(exp_matrix_original)
    exp_matrix_original = pca_original.transform(exp_matrix_original)

    # Dimensionality reduction
    sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=use_highly_variable)

    # Spatial graph construction
    spatial_coords = adata.obsm['spatial']
    neighbor_model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(spatial_coords)
    neighbor_graph = neighbor_model.kneighbors_graph(spatial_coords)

    # Similarity matrix calculation
    sim_matrix = np.exp(2 - cosine_distances(adata.obsm['X_pca'])) - 1
    conn_matrix = neighbor_graph.T.toarray() * sim_matrix

    # Reconstruction
    expression_data = adata.X.toarray() if issparse(adata.X) else adata.X
    reconst_expression = (
        alpha * np.matmul(conn_matrix / np.sum(conn_matrix, axis=0, keepdims=True), expression_data) 
        + expression_data
    )

    # Store results
    smooth_transition_graph = conn_matrix / np.sum(conn_matrix, axis=0, keepdims=True)
    adata.X = csr_matrix(reconst_expression)
    adata.layers['log1p_augmented'] = adata.X

    # Process augmented data
    exp_matrix_augmented = adata.to_df(layer='log1p_augmented')[hvg_genes].to_numpy()
    exp_matrix_augmented = pca_original.transform(exp_matrix_augmented)

    # Clean up and store parameters
    del adata.obsm['X_pca']
    
    adata.uns['reconstruct_spatial_expression'] = {
        'params': {
            'alpha': alpha,
            'n_neighbors': n_neighbors,
            'n_pcs': n_pcs,
            'use_highly_variable': use_highly_variable,
            'normalize_total': normalize_total
        }
    }

    return (
        adata if copy else None,
        exp_matrix_original,
        exp_matrix_augmented,
        smooth_transition_graph
    )
    


def apply_mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='MLN2SVG', random_seed=100):
    """Clustering using the mclust algorithm in R via rpy2"""
    import numpy as np
    import pandas as pd
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    from rpy2.robjects import numpy2ri

    # Set random seeds
    np.random.seed(random_seed)
    robjects.r['set.seed'](random_seed)

    # Activate numpy-to-R conversion
    numpy2ri.activate()

    # Load R package and Mclust function
    robjects.r.library("mclust")
    rmclust = robjects.r['Mclust']

    # Run mclust on the selected obsm
    r_matrix = numpy2ri.numpy2rpy(adata.obsm[used_obsm])
    res = rmclust(r_matrix, num_cluster, modelNames)

    # Extract classification
    mclust_res = np.array(res[-2])  # classification labels

    # Correct way to assign categorical labels
    adata.obs['mclust'] = pd.Series(mclust_res.astype(int), index=adata.obs.index, dtype='category')

    return adata


def refine_spatial_domains(y_pred, coord, n_neighbors=6):
    """
    Refines predicted spatial domains by smoothing labels using the majority vote
    of neighboring spots.

    Parameters:
    -----------
    y_pred : pd.Series
        Predicted domain labels for each spatial spot (indexed by spot).
    coord : np.ndarray or pd.DataFrame
        Spatial coordinates of the spots (shape: [n_spots, 2]).
    n_neighbors : int, optional (default=6)
        Number of spatial neighbors to consider for smoothing.

    Returns:
    --------
    pd.Categorical
        Refined spatial domain labels after neighborhood-based correction.
    """

    # Find nearest neighbors (excluding self)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coord)
    _, indices = nbrs.kneighbors(coord)
    neighbor_indices = indices[:, 1:]  # remove self (first column)

    # Prepare refined labels
    y_refined = pd.Series(index=y_pred.index, dtype='object')

    for i in range(len(y_pred)):
        neighbor_labels = y_pred.iloc[neighbor_indices[i]]
        label_counts = neighbor_labels.value_counts()

        # Replace label if it's not consistent with neighbors
        if (label_counts.get(y_pred.iloc[i], 0) < n_neighbors / 2) and (label_counts.max() > n_neighbors / 2):
            y_refined.iloc[i] = label_counts.idxmax()
        else:
            y_refined.iloc[i] = y_pred.iloc[i]

    # Convert to ordered categorical for consistency
    y_refined = pd.Categorical(
        y_refined.astype(str),
        categories=natsorted(map(str, pd.unique(y_refined))),
        )

    return y_refined


def plot_umap_paga(adata, cluster_key='mln2svg_clust', ari_score=None, title_prefix='MLN2SVG'):
    import matplotlib.pyplot as plt
    import scanpy as sc

    adata = adata[adata.obs[cluster_key].notna()].copy()
    sc.tl.umap(adata)
    sc.tl.paga(adata, groups=cluster_key)

    plt.rcParams["figure.figsize"] = (5, 5)
    plt.rcParams["figure.facecolor"] = "white"

    title = f"{title_prefix} (ARI={ari_score:.2f})" if ari_score is not None else title_prefix

    sc.pl.paga_compare(
        adata,
        legend_fontsize=10,
        frameon=False,
        size=50,
        title=title,
        legend_fontoutline=2,
        show=False
    )
    plt.show()
    
custom_palette = [
    "#1F8FFF",  # Deep neural blue (Layer I)
    "#FF6E3A",  # Pyramidal orange (Layer II/III)
    "#20C997",  # Mitochondrial green (Layer IV)
    "#A45EFA",  # Axon purple (Layer V)
    "#FFD43B",  # Glial gold (WM/VI)
    "#466129D1",  # Dendrite teal
    "#E01A1A",  # Blood vessel coral
    "#9775FA",  # Synaptic violet
    "#51CF66",  # Inhibitory green
    "#F783AC",  # Microglia pink
    "#748FFC",  # Oligo blue
    "#E8590C",  # Excitatory rust
    "#DB4C6C", "#9E7A7A", "#554236", "#AF5F3C", "#93796C",
    "#7D6632", "#DAB370", "#877F6C", "#268785"
]


        
@numba.njit("f4(f4[:], f4[:])")
def compute_euclidean_distance(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i] - t2[i])**2
    return np.sqrt(sum)


def pairwise_euclidean_distances(X):
    n = X.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = compute_euclidean_distance(X[i], X[j])
    return adj


def spatial_adjacency_matrix(x, y):
    print("Calculating adjacency matrix using xy coordinates...")
    X = np.array([x, y]).T.astype(np.float32)
    return pairwise_euclidean_distances(X)


def identify_cluster_markers(
    adata,
    cluster_of_interest,
    neighbor_clusters,
    cluster_column,
    include_neighbors=True,
    log=False
):
    if include_neighbors:
        neighbor_clusters = neighbor_clusters + [cluster_of_interest]
        adata = adata[adata.obs[cluster_column].isin(neighbor_clusters)]
    else:
        adata = adata.copy()

    adata.var_names_make_unique()
    adata.obs["target"] = ((adata.obs[cluster_column] == cluster_of_interest) * 1).astype('category')

    sc.tl.rank_genes_groups(
        adata,
        groupby="target",
        reference="rest",
        n_genes=adata.shape[1],
        method='wilcoxon'
    )

    pvals_adj = [i[0] for i in adata.uns['rank_genes_groups']["pvals_adj"]]
    genes = [i[1] for i in adata.uns['rank_genes_groups']["names"]]

    if issparse(adata.X):
        expression_df = pd.DataFrame(adata.X.A)
    else:
        expression_df = pd.DataFrame(adata.X)

    expression_df.index = adata.obs["target"].tolist()
    expression_df.columns = adata.var.index.tolist()
    expression_df = expression_df.loc[:, genes]

    mean_expression = expression_df.groupby(level=0).mean()
    expression_bool = expression_df.astype(bool)
    expression_frequency = expression_bool.groupby(level=0).sum() / expression_bool.groupby(level=0).count()

    if log:
        fold_change = np.exp((mean_expression.loc[1] - mean_expression.loc[0]).values)
    else:
        fold_change = (mean_expression.loc[1] / (mean_expression.loc[0] + 1e-9)).values

    results = pd.DataFrame({
        'genes': genes,
        'in_group_fraction': expression_frequency.loc[1].tolist(),
        'out_group_fraction': expression_frequency.loc[0].tolist(),
        'in_out_group_ratio': (expression_frequency.loc[1] / expression_frequency.loc[0]).tolist(),
        'in_group_mean_exp': mean_expression.loc[1].tolist(),
        'out_group_mean_exp': mean_expression.loc[0].tolist(),
        'fold_change': fold_change.tolist(),
        'pvals_adj': pvals_adj
    })

    return results


def calculate_neighbor_count(
    cluster_id,
    cell_ids,
    x_coords,
    y_coords,
    cluster_labels,
    radius,
    distance_matrix=None
):
    if distance_matrix is None:
        distance_matrix = spatial_adjacency_matrix(x=x_coords, y=y_coords)

    cell_data = pd.DataFrame({
        'cell_id': cell_ids,
        'x': x_coords,
        'y': y_coords,
        'cluster': cluster_labels
    })
    cell_data.index = cell_data['cell_id']

    target_cells = cell_data[cell_data["cluster"] == cluster_id]
    neighbor_counts = []

    for _, cell in target_cells.iterrows():
        x, y = cell["x"], cell["y"]
        neighborhood_cells = cell_data[((cell_data["x"] - x)**2 + (cell_data["y"] - y)**2) <= (radius**2)]
        neighbor_counts.append(neighborhood_cells.shape[0])

    return np.mean(neighbor_counts)


def find_optimal_radius(
    cluster_id,
    cell_ids,
    x_coords,
    y_coords,
    cluster_labels,
    min_radius,
    max_radius,
    min_neighbors=8,
    max_neighbors=15,
    max_iterations=100
):
    run = 0
    low_count = calculate_neighbor_count(cluster_id, cell_ids, x_coords, y_coords, cluster_labels, min_radius)
    high_count = calculate_neighbor_count(cluster_id, cell_ids, x_coords, y_coords, cluster_labels, max_radius)

    if min_neighbors <= low_count <= max_neighbors:
        print(f"Recommended radius = {min_radius}")
        return min_radius
    elif min_neighbors <= high_count <= max_neighbors:
        print(f"Recommended radius = {max_radius}")
        return max_radius
    elif low_count > max_neighbors:
        print("Try smaller min_radius.")
        return None
    elif high_count < min_neighbors:
        print("Try bigger max_radius.")
        return None

    while (low_count < min_neighbors) and (high_count > max_neighbors):
        run += 1
        print(f"Run {run}: radius [{min_radius}, {max_radius}], neighbors [{low_count}, {high_count}]")

        if run > max_iterations:
            print("Exact radius not found, closest values are:\n"
                  f"radius={min_radius}: neighbors={low_count}\n"
                  f"radius={max_radius}: neighbors={high_count}")
            return None

        mid_radius = (min_radius + max_radius) / 2
        mid_count = calculate_neighbor_count(cluster_id, cell_ids, x_coords, y_coords, cluster_labels, mid_radius)

        if min_neighbors <= mid_count <= max_neighbors:
            print(f"Recommended radius = {mid_radius}, neighbors={mid_count}")
            return mid_radius

        if mid_count < min_neighbors:
            min_radius = mid_radius
            low_count = mid_count
        elif mid_count > max_neighbors:
            max_radius = mid_radius
            high_count = mid_count


def identify_spatial_neighbors(
    target_cluster: int,
    cell_ids: list,
    x_coords: list,
    y_coords: list,
    cluster_labels: list,
    radius: float,
    min_ratio: float = 0.5
) -> list:
    """
    Identify neighboring clusters based on spatial proximity and minimum association ratio.
    
    Args:
        target_cluster: ID of the target cluster
        cell_ids: List of cell identifiers
        x_coords: List of x coordinates
        y_coords: List of y coordinates
        cluster_labels: Cluster labels for all cells
        radius: Search radius for neighbors
        min_ratio: Minimum fraction of cells in a cluster that must be within radius
        
    Returns:
        List of neighboring cluster IDs sorted by association strength
    """
    # Calculate cluster sizes
    cluster_counts = Counter(cluster_labels)
    
    # Create spatial dataframe
    spatial_data = pd.DataFrame({
        'cell_id': cell_ids,
        'x': x_coords,
        'y': y_coords,
        'cluster': cluster_labels
    }).set_index('cell_id')
    
    # Filter target cluster cells
    target_cells = spatial_data[spatial_data["cluster"] == target_cluster]
    neighbor_counts = Counter()
    neighbor_stats = []
    
    # Count neighbors for each target cell
    for _, cell in target_cells.iterrows():
        x, y = cell["x"], cell["y"]
        neighbors = spatial_data[((spatial_data["x"]-x)**2 + (spatial_data["y"]-y)**2) <= (radius**2)]
        neighbor_stats.append(neighbors.shape[0])
        neighbor_counts.update(neighbors["cluster"])
    
    # Remove target cluster from neighbors
    del neighbor_counts[target_cluster]
    original_counts = neighbor_counts.copy()
    
    # Apply ratio filter
    filtered_neighbors = [
        (cluster, count) 
        for cluster, count in neighbor_counts.items() 
        if count > (min_ratio * cluster_counts[cluster])
    ]
    filtered_neighbors.sort(key=lambda x: -x[1])  # Sort by count descending
    
    # Log results
    avg_neighbors = np.mean(neighbor_stats)
    print(f"Radius={radius:.2f}, average neighbors per spot: {avg_neighbors:.1f}")
    print(f"Cluster {target_cluster} potential neighbors:")
    for cluster, count in filtered_neighbors:
        print(f"Domain {cluster}: {count} cells ({(count/cluster_counts[cluster]):.1%})")
    
    # Handle no neighbors case
    if not filtered_neighbors:
        backup_neighbor = [original_counts.most_common(1)[0][0]]
        print(f"No neighbors found. Returning most frequent: {backup_neighbor}")
        return backup_neighbor
    
    return [cluster for cluster, _ in filtered_neighbors]

def find_neighbor_domains(target_cluster, n_adata):
    spatial_coords = n_adata.obsm["spatial"]
    x_coords, y_coords = spatial_coords[:, 0], spatial_coords[:, 1]

    distance_matrix = spatial_adjacency_matrix(x=x_coords, y=y_coords)

    start = np.quantile(distance_matrix[distance_matrix != 0], q=0.001)
    end = np.quantile(distance_matrix[distance_matrix != 0], q=0.1)

    optimal_radius = find_optimal_radius(
        cluster_id=target_cluster,
        cell_ids=n_adata.obs.index.tolist(),
        x_coords=x_coords,
        y_coords=y_coords,
        cluster_labels=n_adata.obs["mclust"].tolist(),
        min_radius=start,
        max_radius=end,
        min_neighbors=10,
        max_neighbors=14,
        max_iterations=100
    )
    
#         # As used in find_neighbor_domains():
# neighbors = identify_spatial_neighbors(
#     target_cluster=target_cluster,
#     cell_ids=cell_ids,
#     x_coords=x_coords,
#     y_coords=y_coords,
#     cluster_labels=cluster_labels,
#     radius=radius,
#     min_ratio=1/2
# )
    neighbor_domains = identify_spatial_neighbors(
        target_cluster=target_cluster,
        cell_ids=n_adata.obs.index.tolist(),
        x_coords=x_coords,
        y_coords=y_coords,
        cluster_labels=n_adata.obs["mclust"].tolist(),
        radius=optimal_radius,
        min_ratio=1/2
    )

    return neighbor_domains


def find_spatial_markers(
    adata,
    adjacency_matrix,
    cluster_column="mclust",
    min_in_group_fraction=0.8,
    min_in_out_group_ratio=1.0,
    min_fold_change=1.2,
    pval_threshold=0.05,
    log_transform=True
):
    filtered_markers = []

    for target_domain, neighbors in adjacency_matrix.items():
        if not neighbors:
            continue

        de_results = identify_cluster_markers(
            adata=adata,
            cluster_of_interest=target_domain,
            neighbor_clusters=neighbors,
            cluster_column=cluster_column,
            include_neighbors=True,
            log=log_transform
        )

        de_results = de_results[
            (de_results["pvals_adj"] < pval_threshold) &
            (de_results["in_out_group_ratio"] > min_in_out_group_ratio) &
            (de_results["in_group_fraction"] > min_in_group_fraction) &
            (de_results["fold_change"] > min_fold_change)
        ].sort_values(by="in_group_fraction", ascending=False)

        de_results["target_domain"] = target_domain
        de_results["neighbors"] = str(neighbors)

        print(f"SVGs for domain {target_domain}: {de_results['genes'].tolist()}")
        print(f"Number of significant genes: {len(de_results)}")

        filtered_markers.append(de_results)

    return filtered_markers


def compute_second_order_adjacency(adjacency_matrix):
    second_order_adj = dict()

    for domain in list(adjacency_matrix.keys()):
        first_order_neighbors = adjacency_matrix[domain]
        second_order_neighbors = set()

        for neighbor in first_order_neighbors:
            for second_neighbor in adjacency_matrix[neighbor]:
                if domain == second_neighbor or second_neighbor in first_order_neighbors:
                    continue
                for third_neighbor in adjacency_matrix[second_neighbor]:
                    if domain == third_neighbor or third_neighbor in first_order_neighbors:
                        continue
                    second_order_neighbors.add(third_neighbor)

        second_order_adj[domain] = second_order_neighbors

    for domain in list(second_order_adj.keys()):
        for neighbor in list(second_order_adj[domain]):
            second_order_adj[neighbor].add(domain)

    for domain in list(second_order_adj.keys()):
        second_order_adj[domain] = list(second_order_adj[domain])

    print("Multi-level (second-order) adjacency matrix computed")
    print(second_order_adj)
    print(f"Result type: {type(second_order_adj)}")
    return second_order_adj

def find_multi_order_svg(
    adata,
    adjacency_dict,
    cluster_column="mclust",
    min_in_group_fraction=0.8,
    min_in_out_group_ratio=1.0,
    min_fold_change=1.2,
    pval_threshold=0.05,
    log_transform=True
    ):
    svg_results = []

    for target_domain, neighbors in adjacency_dict.items():
        if not neighbors:
            continue

        de_results = identify_cluster_markers(
            adata=adata,
            cluster_of_interest=target_domain,
            neighbor_clusters=neighbors,
            cluster_column=cluster_column,
            include_neighbors=True,
            log=log_transform
        )

        filtered_results = de_results[
            (de_results["pvals_adj"] < pval_threshold) &
            (de_results["in_out_group_ratio"] > min_in_out_group_ratio) &
            (de_results["in_group_fraction"] > min_in_group_fraction) &
            (de_results["fold_change"] > min_fold_change)
        ].sort_values(by="in_group_fraction", ascending=False)

        filtered_results["target_domain"] = target_domain
        filtered_results["neighbors"] = str(neighbors)

        print(f"SVGs for domain {target_domain}: {filtered_results['genes'].tolist()}")
        print(f"Number of significant genes: {len(filtered_results)}")

        svg_results.append(filtered_results)

    return svg_results


def combine_spatial_markers(
    spatial_markers_1: list,
    spatial_markers_2: list,
    sort_column: str = 'pvals_adj',
    merge_method: str = 'union'
) -> list:
    """
    Combines SVG results from multiple adjacency orders while preserving all data.
    
    Args:
        spatial_markers_1: First list of SVG DataFrames (e.g., from first-order adj)
        spatial_markers_2: Second list of SVG DataFrames (e.g., from second-order adj)
        sort_column: Column to sort final results by (default: 'pvals_adj')
        merge_method: 'union' (keep all genes) or 'intersection' (only common genes)
        
    Returns:
        List of merged DataFrames, one per spatial domain
        
    Example:
        combined_results = combine_spatial_markers(first_order_results, second_order_results)
    """
    from collections import defaultdict
    
    # Validate input
    if not all(isinstance(x, pd.DataFrame) for lst in [spatial_markers_1, spatial_markers_2] for x in lst):
        raise ValueError("All inputs must be lists of pandas DataFrames")
    
    # Initialize merging structure
    domain_dict = defaultdict(list)
    
    # Combine all DataFrames by domain
    for df in spatial_markers_1 + spatial_markers_2:
        try:
            domain = str(df['target_domain'].iloc[0])  # Using standardized column name
            domain_dict[domain].append(df)
        except (KeyError, AttributeError) as e:
            print(f"Skipping invalid DataFrame: {str(e)}")
            continue
    
    # Process each domain
    merged_results = []
    for domain, dfs in domain_dict.items():
        # Concatenate while preserving all columns
        combined = pd.concat(dfs, axis=0)
        
        # Handle duplicates based on merge method
        if merge_method == 'union':
            combined = combined.drop_duplicates(subset=['genes'], keep='first')
        elif merge_method == 'intersection':
            gene_counts = combined['genes'].value_counts()
            common_genes = gene_counts[gene_counts == len(dfs)].index
            combined = combined[combined['genes'].isin(common_genes)]
        
        # Sort by significance
        combined = combined.sort_values(sort_column)
        
        # Add metadata
        combined['source_datasets'] = len(dfs)
        combined['merge_method'] = merge_method
        
        merged_results.append(combined)
    
    # Print summary
    print(f"Merged results for {len(merged_results)} domains:")
    print(f"- First-order domains: {len(spatial_markers_1)}")
    print(f"- Second-order domains: {len(spatial_markers_2)}")
    print(f"- Merge method: {merge_method}")
    
    return merged_results


def plot_spatial_gene_expression(adata, domain_data, cluster_col='mln2svg_clust'):
    """Plot spatial gene expression for top genes in each domain."""
    
    # Custom colormap
    color_self = LinearSegmentedColormap.from_list(
        'pink_green', ['#3AB370', "#EAE7CC", "#FD1593"], N=256
    )

    # Get spatial coordinates
    spatial_coords = adata.obsm['spatial']
    if spatial_coords.shape[1] != 2:
        spatial_coords = spatial_coords[:, :2]

    # Ensure domain_data is a list
    domain_dfs = domain_data if isinstance(domain_data, list) else [domain_data]

    for domain_df in domain_dfs:
        if len(domain_df) == 0:
            continue

        # Identify top gene per domain
        top_gene_row = domain_df.sort_values('pvals_adj').iloc[0]
        g = top_gene_row['genes']
        domain = top_gene_row['target_domain']

        # Aspect ratio setup
        y_max, x_max = spatial_coords.max(axis=0)
        fig, ax = plt.subplots(figsize=(6, 6 * (y_max / x_max)))

        # Extract gene expression values
        if g in adata.var_names:
            gene_idx = adata.var_names.get_loc(g)
            exp_values = (
                adata.X[:, gene_idx].toarray().flatten()
                if issparse(adata.X)
                else adata.X[:, gene_idx]
            )
        else:
            print(f"⚠️ Gene {g} not found in adata.var_names.")
            continue

        # Scatter plot
        sc = ax.scatter(
            x=spatial_coords[:, 0],
            y=spatial_coords[:, 1],
            c=exp_values,
            cmap=color_self,
            s=20,
            alpha=0.8,
            edgecolor='none'
        )

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc, cax=cax, label='Expression')

        # Title and labels
        ax.set_title(f"Domain {domain}: {g}")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")

        # Highlight domain region
        domain_mask = adata.obs[cluster_col] == str(domain)
        if domain_mask.any():
            domain_coords = spatial_coords[domain_mask]
            x_center, y_center = domain_coords.mean(axis=0)
            ax.text(
                x_center, y_center, f"Domain {domain}",
                fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
            )

        # Style adjustments
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

        plt.tight_layout()
        plt.show()
        
        

def generate_gene_clouds(domain_data_list, output_dir, colormap='viridis', max_words=100, show=True):
    """
    Generate, display, and save word clouds for gene expression domains.

    Parameters
    ----------
    domain_data_list : list
        A list of DataFrames, each containing gene info for one domain.
    output_dir : str
        Directory path where word cloud images will be saved.
    colormap : str, optional
        Matplotlib colormap for the word cloud. Default is 'viridis'.
    max_words : int, optional
        Maximum number of words to include in each cloud. Default is 100.
    show : bool, optional
        Whether to display each word cloud inline (useful for notebooks).
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving gene clouds to: {output_dir}")

    # Iterate through all domain-specific DataFrames
    for filtered_info in domain_data_list:
        if len(filtered_info) == 0:
            continue

        # Extract domain name
        target_domain = filtered_info['target_domain'].iloc[0]

        # Extract gene names
        genes = filtered_info['genes'].tolist()

        # Determine weights (fold change or similar metric)
        if 'fold_change' in filtered_info.columns:
            weights = filtered_info['fold_change'].tolist()
        elif 'logfoldchanges' in filtered_info.columns:
            weights = filtered_info['logfoldchanges'].tolist()
        elif 'avg_logFC' in filtered_info.columns:
            weights = filtered_info['avg_logFC'].tolist()
        else:
            weights = [1] * len(genes)

        # Create dictionary mapping genes to their weights
        wordcloud_data = dict(zip(genes, weights))

        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap=colormap,
            max_words=max_words,
            min_font_size=10
        ).generate_from_frequencies(wordcloud_data)

        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Gene Cloud for Domain {target_domain}', fontsize=16)

        # Save to file
        save_path = os.path.join(output_dir, f"GeneCloud_Domain_{target_domain}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

        # Show inline if enabled
        if show:
            plt.show()
        else:
            plt.close()

    print("✅ All gene clouds saved and displayed successfully!")

