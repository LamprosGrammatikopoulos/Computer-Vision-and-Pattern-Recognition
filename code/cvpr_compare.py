# ------------------------------------------
# | Full Name: Lampros Grammatikopoulos    |
# | University email: lg01302@surrey.ac.uk |
# | University number: 6918674             |
# ------------------------------------------

import numpy as np

def cvpr_compare(F1, F2, metric='euclidean', inv_cov=None, p=3):
    """
    Compare two feature descriptors using various distance metrics.
    
    Parameters
    ----------
    F1, F2 : numpy arrays
        Feature descriptors to compare
    method : str
        Distance metric to use. Supported options:
            'euclidean'     - L2 norm (Euclidean distance)
            'manhattan'     - L1 norm (Manhattan distance)
            'chebyshev'     - L∞ norm (maximum absolute difference)
            'cosine'        - 1 minus cosine similarity
            'canberra'      - Canberra distance
            'mahalanobis'   - Mahalanobis distance (requires inv_cov)
            'minkowski'     - Generalized Lp norm (controlled by p)
            'chi-square'    - Chi-square distance (commonly used for histograms)
            'intersection'  - Histogram intersection distance (1 - intersection)
            'bhattacharyya' - Bhattacharyya distance (for probability distributions)
            'hellinger'     - Hellinger distance (related to Bhattacharyya)
            'correlation'   - Correlation distance (OpenCV CV_COMP_CORREL)
    inv_cov : numpy array, optional
        Inverse covariance matrix (required for Mahalanobis distance)
    p : float, optional
        Order of the Minkowski distance (default = 3)
    
    Returns
    -------
    dst : float
        Distance between F1 and F2
    """
    F1, F2 = np.asarray(F1), np.asarray(F2)
    
    if metric == 'euclidean':
        # L2 (Euclidean) distance
        dst = np.linalg.norm(F1 - F2)
        # dst = np.sqrt(np.sum((F1 - F2) ** 2))
    elif metric == 'manhattan':
        # L1 (Manhattan) distance
        dst = np.sum(np.abs(F1 - F2))
    elif metric == 'chebyshev':
        dst = np.max(np.abs(F1 - F2))
    elif metric == 'cosine':
        num = np.dot(F1, F2)
        denom = np.linalg.norm(F1) * np.linalg.norm(F2)
        dst = 1.0 - (num / denom) if denom != 0 else 1.0
    elif metric == 'canberra':
        num = np.abs(F1 - F2)
        denom = np.abs(F1) + np.abs(F2)
        denom[denom == 0] = 1e-12  # prevent division by zero
        dst = np.sum(num / denom)
    elif metric == 'mahalanobis':
        if inv_cov is None:
            raise ValueError("Mahalanobis distance requires inv_cov matrix")
        diff = F1 - F2
        # Faster: compute diff @ inv_cov first (reduces to vector), then dot with diff
        dst = np.sqrt(np.dot(diff, np.dot(inv_cov, diff)))
    elif metric == 'minkowski':
        # Generalized Lp norm (p can be any positive real number)
        diff = np.abs(F1 - F2) ** p
        dst = np.sum(diff) ** (1.0 / p)
    elif metric == 'chi-square':
        # Chi-square distance - standard for histogram comparison
        epsilon = 1e-10
        denominator = F1 + F2 + epsilon
        numerator = (F1 - F2) ** 2
        dst = 0.5 * np.sum(numerator / denominator)
        
    elif metric == 'intersection':
        # Histogram intersection (return distance, not similarity)
        intersection = np.sum(np.minimum(F1, F2))
        dst = 1.0 - intersection
        
    elif metric == 'bhattacharyya':
        epsilon = 1e-10
        bc_coeff = np.sum(np.sqrt(F1 * F2))
        bc_coeff = np.clip(bc_coeff, epsilon, 1.0)
        dst = -np.log(bc_coeff)
        
    elif metric == 'hellinger':
        dst = np.sqrt(0.5 * np.sum((np.sqrt(F1) - np.sqrt(F2)) ** 2))
        
    elif metric == 'correlation':
        # Correlation distance (OpenCV CV_COMP_CORREL)
        # d(H1, H2) = Σ(H1(I) - H̄1)(H2(I) - H̄2) / sqrt(Σ(H1(I) - H̄1)² Σ(H2(I) - H̄2)²)
        H1_mean = np.mean(F1)
        H2_mean = np.mean(F2)
        
        H1_centered = F1 - H1_mean
        H2_centered = F2 - H2_mean
        
        numerator = np.sum(H1_centered * H2_centered)
        denominator = np.sqrt(np.sum(H1_centered ** 2) * np.sum(H2_centered ** 2))
        
        if denominator != 0:
            correlation = numerator / denominator
        else:
            correlation = 0.0
        
        # Convert correlation to distance (higher correlation = lower distance)
        # OpenCV returns the correlation value itself, but for distance we use 1 - correlation
        dst = 1.0 - correlation
        
    else:
        raise ValueError(f"Unknown distance metric: {metric}")
    
    return float(dst)