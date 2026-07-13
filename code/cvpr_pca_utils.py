# ------------------------------------------
# | Full Name: Lampros Grammatikopoulos    |
# | University email: lg01302@surrey.ac.uk |
# | University number: 6918674             |
# ------------------------------------------

import numpy as np
import os
import pickle
from scipy import linalg as sp_linalg

def compute_pca_projection(descriptors, n_components=None, variance_threshold=0.95):
    """
    PCA using teacher's Eigenmodel approach (EVD method).
    Optimized for speed with scipy.linalg.
    
    Eigenmodel = (μ, U, V) where:
    - μ: mean
    - U: eigenvectors 
    - V: eigenvalues
    
    Steps:
    1) Arrange data in matrix (samples as rows)
    2) Calculate mean μ
    3) Subtract mean: p = x - μ
    4) Calculate covariance: C = p.T @ p / (n-1)
    5) Eigenvalue decomposition: [U, V] = eig(C)
    
    Parameters:
    -----------
    descriptors : numpy array (n_samples, n_features)
        Original high-dimensional descriptors
    n_components : int or None
        Number of components to keep. If None, use variance_threshold
    variance_threshold : float
        Keep enough components to explain this fraction of variance (0-1)
    
    Returns:
    --------
    pca_model : dict
        Eigenmodel containing μ, U, V
    projected : numpy array (n_samples, n_components)
        Projected descriptors
    explained_variance : float
        Total variance explained by kept components
    """
    
    print(f"\nOriginal descriptor shape: {descriptors.shape}")
    print(f"Original dimensionality: {descriptors.shape[1]}")
    print("Building Eigenmodel using EVD (Teacher's method, optimized)...")
    
    n_samples, n_features = descriptors.shape
    
    # Step 1: Data already arranged in matrix (n_samples × n_features)
    
    # Step 2: Calculate the mean μ
    print("Step 1: Computing mean μ...")
    mu = np.mean(descriptors, axis=0)
    
    # Step 3: Subtract mean from data: p = x - μ
    print("Step 2: Subtracting mean (p = x - μ)...")
    p = descriptors - mu
    
    # Step 4: Calculate covariance C = p.T @ p / (n-1)
    print("Step 3: Computing covariance C = p.T @ p...")
    
    # OPTIMIZATION: For high dimensions, use reduced-rank approach
    # Since n_samples < n_features, compute p @ p.T instead (much smaller!)
    if n_samples < n_features:
        print(f"  Using reduced computation: {n_samples} < {n_features}")
        print(f"  Computing {n_samples}×{n_samples} matrix instead of {n_features}×{n_features}")
        
        # Compute p @ p.T (n_samples × n_samples) - much faster!
        C_small = (p @ p.T) / (n_samples - 1)
        
        # Eigendecomposition of small matrix
        print("Step 4: Computing eigendecomposition [U_small, V] = eig(C_small)...")
        eigenvalues, U_small = sp_linalg.eigh(C_small)
        
        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        U_small = U_small[:, idx]
        
        # Remove near-zero eigenvalues
        positive_idx = eigenvalues > 1e-10
        eigenvalues = eigenvalues[positive_idx]
        U_small = U_small[:, positive_idx]
        
        # Recover full eigenvectors: U = p.T @ U_small / sqrt(eigenvalues * (n-1))
        print("Step 5: Recovering full eigenvectors...")
        U = p.T @ U_small
        # Normalize eigenvectors
        for i in range(U.shape[1]):
            U[:, i] = U[:, i] / np.linalg.norm(U[:, i])
        
    else:
        # Standard approach for low dimensions
        print(f"  Computing full covariance matrix...")
        C = (p.T @ p) / (n_samples - 1)
        
        print(f"  Covariance matrix shape: {C.shape}")
        
        # Step 5: Eigenvalue decomposition [U, V] = eig(C)
        print("Step 4: Computing eigendecomposition [U, V] = eig(C)...")
        eigenvalues, U = sp_linalg.eigh(C)
        
        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        U = U[:, idx]
    
    # Handle negative eigenvalues (numerical errors)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Calculate explained variance ratio
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    
    # Determine number of components to keep
    if n_components is None:
        cumsum = np.cumsum(explained_variance_ratio)
        n_components = np.searchsorted(cumsum, variance_threshold) + 1
        n_components = min(n_components, len(eigenvalues))
    
    print(f"Step 5: Keeping {n_components} components...")
    
    # Keep only top n_components
    U_kept = U[:, :n_components]
    V_kept = eigenvalues[:n_components]
    explained_variance_ratio = explained_variance_ratio[:n_components]
    
    # Project data: projected = p @ U
    print("Step 6: Projecting data (projected = p @ U)...")
    projected = p @ U_kept
    
    explained_variance = np.sum(explained_variance_ratio)
    
    print(f"\nEigenmodel created:")
    print(f"  μ (mean): shape {mu.shape}")
    print(f"  U (eigenvectors): shape {U_kept.shape}")
    print(f"  V (eigenvalues): {n_components} values")
    print(f"  Variance explained: {explained_variance:.4f}")
    print(f"  Dimensionality reduction: {n_features} -> {n_components} "
          f"({100*n_components/n_features:.1f}%)")
    
    # Print top eigenvalues
    print(f"\nTop 10 eigenvalues and variance:")
    for i in range(min(10, len(V_kept))):
        cumvar = np.sum(explained_variance_ratio[:i+1])
        print(f"  λ{i+1}: {V_kept[i]:.2f} "
              f"(variance: {explained_variance_ratio[i]:.4f}, cumulative: {cumvar:.4f})")
    
    # Store Eigenmodel (μ, U, V)
    pca_model = {
        'mean': mu,                    # μ
        'eigenvectors': U_kept,        # U
        'eigenvalues': V_kept,         # V
        'components': U_kept.T,        # For compatibility
        'explained_variance_ratio': explained_variance_ratio,
        'n_components': n_components
    }
    
    return pca_model, projected, explained_variance


def compute_covariance_matrix(descriptors, regularization=None):
    """
    Compute covariance matrix and its inverse for Mahalanobis distance.
    Uses diagonal approximation for very high dimensions (>10000).
    """
    
    n_samples, n_features = descriptors.shape
    
    # Center data
    mu = np.mean(descriptors, axis=0)
    p = descriptors - mu
    
    # For very high dimensions, use diagonal covariance (much faster)
    if n_features > 10000:
        print(f"WARNING: {n_features} dimensions is too large for full covariance!")
        print("Using DIAGONAL covariance approximation (assumes features are independent)")
        
        # Diagonal covariance: just the variance of each feature
        # Add regularization
        if regularization is not None and regularization > 0:
            variances = np.var(p, axis=0) + regularization
        else:
            variances = np.var(p, axis=0)
        inv_cov_matrix = np.diag(1.0 / variances)
        cov_matrix = np.diag(variances)
        
        print(f"Using diagonal approximation (much faster)")
        return cov_matrix, inv_cov_matrix
    
    # For lower dimensions (after PCA), compute full covariance
    print(f"Computing full covariance matrix...")
    cov_matrix = (p.T @ p) / (n_samples - 1)
    
    # Use minimal regularization by default
    if regularization is not None:
        if regularization > 0:
            cov_matrix += regularization * np.eye(cov_matrix.shape[0])
        # If regularization is 0 or None, use very small value for numerical stability
        else:
            cov_matrix += 1e-8 * np.eye(cov_matrix.shape[0])
    else:
        # Default: minimal regularization for numerical stability only
        cov_matrix += 1e-8 * np.eye(cov_matrix.shape[0])
    
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    
    # Fast inverse using scipy
    try:
        inv_cov_matrix = sp_linalg.inv(cov_matrix, check_finite=False)
        print("Successfully computed inverse covariance matrix")
    except np.linalg.LinAlgError:
        print("Warning: Using pseudo-inverse")
        inv_cov_matrix = sp_linalg.pinv(cov_matrix, check_finite=False)
    
    return cov_matrix, inv_cov_matrix


def save_pca_model(pca_model, inv_cov, filepath):
    """Save PCA Eigenmodel"""
    data = {
        'pca_model': pca_model,
        'inv_cov': inv_cov
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Eigenmodel saved to {filepath}")


def load_pca_model(filepath):
    """Load PCA Eigenmodel"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Eigenmodel loaded from {filepath}")
    return data['pca_model'], data['inv_cov']