# ------------------------------------------
# | Full Name: Lampros Grammatikopoulos    |
# | University email: lg01302@surrey.ac.uk |
# | University number: 6918674             |
# ------------------------------------------

"""
Optimized script for comprehensive PCA and distance metric comparisons.
mAP calculation using full PR curve + keeping k-based metrics
"""

import os
import re
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from cvpr_pca_utils import compute_pca_projection, compute_covariance_matrix

def get_category_from_filename(filename):
    """Extract category from filename"""
    basename = os.path.splitext(os.path.basename(filename))[0]
    return basename.split('_')[0]

def load_all_descriptors(descriptor_subfolder):
    """Load all descriptors for a specific subfolder"""
    ALLFEAT = []
    ALLFILES = []
    
    desc_dir = os.path.join(DESCRIPTOR_FOLDER, descriptor_subfolder)
    for filename in sorted(os.listdir(desc_dir)):
        if filename.endswith('.mat'):
            mat_path = os.path.join(desc_dir, filename)
            try:
                img_data = sio.loadmat(mat_path)
                if 'Hist' in img_data:
                    ALLFILES.append(mat_path)
                    descriptor = img_data['Hist'].flatten()
                    ALLFEAT.append(descriptor)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return np.array(ALLFEAT), ALLFILES

def compute_distance_matrix_vectorized(ALLFEAT, distance_metric, inv_cov=None):
    """
    Compute full distance matrix using vectorized operations.
    Much faster than nested loops for Euclidean and Mahalanobis.
    """
    n = len(ALLFEAT)
    
    if distance_metric == 'euclidean':
        dist_matrix = cdist(ALLFEAT, ALLFEAT, metric='euclidean')
    
    elif distance_metric == 'mahalanobis':
        if inv_cov is None:
            raise ValueError("inv_cov required for Mahalanobis")
        
        X_inv_cov = ALLFEAT @ inv_cov
        diag_terms = np.sum(ALLFEAT * X_inv_cov, axis=1)
        cross_terms = X_inv_cov @ ALLFEAT.T
        dist_matrix_sq = diag_terms[:, None] + diag_terms[None, :] - 2 * cross_terms
        dist_matrix_sq = np.maximum(dist_matrix_sq, 0)
        dist_matrix = np.sqrt(dist_matrix_sq)
    
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    np.fill_diagonal(dist_matrix, np.inf)
    return dist_matrix

def evaluate_experiment(ALLFEAT, ALLFILES, distance_metric, inv_cov=None, k_values=[1, 5, 10, 15, 20]):
    """
    Fast evaluation with BOTH:
    - K-based metrics (P@K[1,5,10,15,20], R@K[1,5,10,15,20])
    - mAP@K[1,max] from full PR curve
    """
    print("  Computing distance matrix...", end='', flush=True)
    dist_matrix = compute_distance_matrix_vectorized(ALLFEAT, distance_metric, inv_cov)
    print(" Done!")
    
    print("  Sorting distances...", end='', flush=True)
    sorted_indices = np.argsort(dist_matrix, axis=1)
    print(" Done!")
    
    # Pre-compute categories
    categories = np.array([get_category_from_filename(os.path.basename(f)) for f in ALLFILES])
    unique_cats, counts = np.unique(categories, return_counts=True)
    cat_to_count = dict(zip(unique_cats, counts))
    
    # Storage for both metrics
    all_precisions_k = {k: [] for k in k_values}
    all_recalls_k = {k: [] for k in k_values}
    all_ap_scores = []  # For mAP@K[1,max]
    
    max_k = max(k_values)
    
    print(f"  Evaluating {len(ALLFEAT)} queries (computing full PR curves)...")
    for query_idx in range(len(ALLFEAT)):
        if query_idx % 100 == 0:
            print(f"  Query {query_idx}/{len(ALLFEAT)}...", end='\r')
        
        query_category = categories[query_idx]
        total_relevant = cat_to_count[query_category] - 1
        
        # Get ALL retrieved (for full PR curve)
        all_retrieved_indices = sorted_indices[query_idx, :]
        retrieved_categories = categories[all_retrieved_indices]
        is_relevant = (retrieved_categories == query_category)
        
        # === 1. Compute K-based metrics ===
        cumulative_relevant = np.cumsum(is_relevant)
        for k in k_values:
            if k <= len(cumulative_relevant):
                relevant_at_k = cumulative_relevant[k-1]
                precision = relevant_at_k / k
                recall = relevant_at_k / total_relevant if total_relevant > 0 else 0
            else:
                precision = 0
                recall = 0
            all_precisions_k[k].append(precision)
            all_recalls_k[k].append(recall)
        
        # === 2. Compute full PR curve for AP@K[1,max] ===
        if total_relevant > 0:
            # Find where all relevant items are retrieved
            positions = np.arange(1, len(is_relevant) + 1)
            precisions_full = cumulative_relevant / positions
            recalls_full = cumulative_relevant / total_relevant
            
            # Stop at last relevant item
            last_relevant_pos = np.where(cumulative_relevant == total_relevant)[0]
            if len(last_relevant_pos) > 0:
                cutoff = last_relevant_pos[0] + 1
                precisions_full = precisions_full[:cutoff]
                recalls_full = recalls_full[:cutoff]
            
            # Monotone interpolation (precision non-increasing)
            for i in reversed(range(len(precisions_full)-1)):
                precisions_full[i] = max(precisions_full[i], precisions_full[i+1])
            
            # Add starting point
            recalls_full = np.concatenate([[0.0], recalls_full])
            precisions_full = np.concatenate([[precisions_full[0] if len(precisions_full) > 0 else 0.0], precisions_full])
            
            # Compute AP as area under curve
            ap_score = np.trapezoid(precisions_full, recalls_full) if len(recalls_full) > 1 else 0.0
        else:
            ap_score = 0.0
        
        all_ap_scores.append(ap_score)
    
    print()  # New line
    
    # Calculate metrics
    avg_precision_k = {k: np.mean(all_precisions_k[k]) for k in k_values}
    avg_recall_k = {k: np.mean(all_recalls_k[k]) for k in k_values}
    map_at_k = np.mean(list(avg_precision_k.values()))  # Simple average for k-based
    best_map_all = np.mean(all_ap_scores)  # mAP@K[1,max] from full curves
    
    return avg_precision_k, avg_recall_k, map_at_k, best_map_all

def run_all_experiments(descriptor_subfolder):
    """Run all experimental configurations for a specific descriptor"""
    print("="*70)
    print(f"DESCRIPTOR: {descriptor_subfolder}")
    print("="*70)
    
    print("\nLoading descriptors...")
    ALLFEAT_original, ALLFILES = load_all_descriptors(descriptor_subfolder)
    print(f"Loaded {len(ALLFEAT_original)} descriptors")
    print(f"Original dimensionality: {ALLFEAT_original.shape[1]}")
    
    results = []
    k_values = [1, 5, 10, 15, 20]

    for exp_num, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*70}")
        print(f"Experiment {exp_num}/{len(EXPERIMENTS)}: {exp['name']}")
        print(f"{'='*70}")
        
        # Apply PCA if needed
        if exp['pca']:
            print(f"Applying PCA (variance threshold: {exp['variance']})...")
            pca_model, ALLFEAT, explained_var = compute_pca_projection(
                ALLFEAT_original,
                n_components=None,
                variance_threshold=exp['variance']
            )
            n_dims = ALLFEAT.shape[1]
            print(f"  Reduced to {n_dims} dimensions")

            # Save PCA descriptors
            full_save_folder = os.path.join(PCA_DESCRIPTOR_FOLDER, descriptor_subfolder)
            os.makedirs(full_save_folder, exist_ok=True)
            variance_str = str(int(exp['variance'] * 100))
            distance_str = exp['distance']
            save_path = os.path.join(full_save_folder, f"{descriptor_subfolder}_PCA{variance_str}_{distance_str}.mat")
            sio.savemat(save_path, {'ALLFEAT_PCA': ALLFEAT})
            print(f"  Saved PCA descriptors to: {save_path}")
        else:
            print("Using original features (no PCA)")
            ALLFEAT = ALLFEAT_original
            n_dims = ALLFEAT.shape[1]
        
        # Compute inverse covariance
        inv_cov = None
        if exp['distance'] == 'mahalanobis':
            regularization = 1e-6 if exp['pca'] else 1e-8
            print(f"Computing covariance matrix with regularization={regularization:.0e}...")
            cov_matrix, inv_cov = compute_covariance_matrix(ALLFEAT, regularization=regularization)
        
        # Evaluate (now returns 4 values)
        print(f"Evaluating with {exp['distance']} distance...")
        avg_precision, avg_recall, map_at_k, best_map_all = evaluate_experiment(
            ALLFEAT, ALLFILES, exp['distance'], inv_cov, k_values
        )
        
        # Store results
        result = {
            'name': exp['name'],
            'pca': exp['pca'],
            'distance': exp['distance'],
            'dimensions': n_dims,
            'map_at_k': map_at_k,      # Average of P@K[1,5,10,15,20] values
            'best_map_all': best_map_all,       # mAP@K[1,max] from full PR curve
        }
        
        for k in k_values:
            result[f'P@{k}'] = avg_precision[k]
            result[f'R@{k}'] = avg_recall[k]
        
        results.append(result)
        
        print(f"\nResults: mAP@K[1,5,10,15,20] = {map_at_k:.4f}, mAP@K[1,max] = {best_map_all:.4f}, Dims = {n_dims}")
    
    return results, k_values

def plot_comparison_results(results, k_values, descriptor_subfolder):
    """Create comprehensive comparison plots with BOTH mAP metrics"""
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Descriptor: {descriptor_subfolder}', fontsize=16, fontweight='bold', y=0.98)
    
    names = [r['name'] for r in results]
    colors = ['blue' if 'Euclidean' in n else 'red' for n in names]
    
    # 1. mAP@K[1,5,10,15,20] Comparison (original - average of P@k)
    ax1 = plt.subplot(2, 4, 1)
    maps_at_k = [r['map_at_k'] for r in results]
    ax1.barh(range(len(names)), maps_at_k, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('mAP@K[1,5,10,15,20] Score', fontsize=10)
    ax1.set_title('mAP@K[1,5,10,15,20] (Avg of P@K[1,5,10,15,20])', fontsize=11, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=max(maps_at_k), color='green', linestyle='--', linewidth=2, alpha=0.5)
    
    # 2. mAP@K[1,max] Comparison (from full PR curve)
    ax2 = plt.subplot(2, 4, 2)
    maps_all = [r['best_map_all'] for r in results]
    ax2.barh(range(len(names)), maps_all, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel('mAP@K[1,max] Score', fontsize=10)
    ax2.set_title('mAP@K[1,max] (Area under PR curve)', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(x=max(maps_all), color='green', linestyle='--', linewidth=2, alpha=0.5)
    
    # 3. Dimensionality Comparison
    ax3 = plt.subplot(2, 4, 3)
    dims = [r['dimensions'] for r in results]
    ax3.barh(range(len(names)), dims, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=8)
    ax3.set_xlabel('Number of Dimensions', fontsize=10)
    ax3.set_title('Dimensionality Comparison', fontsize=11, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. mAP Comparison (both metrics side by side)
    ax4 = plt.subplot(2, 4, 4)
    x = np.arange(len(names))
    width = 0.35
    ax4.barh(x - width/2, maps_at_k, width, label='mAP@K[1,5,10,15,20]', color='skyblue', alpha=0.7)
    ax4.barh(x + width/2, maps_all, width, label='mAP@K[1,max]', color='orange', alpha=0.7)
    ax4.set_yticks(x)
    ax4.set_yticklabels(names, fontsize=8)
    ax4.set_xlabel('mAP Score', fontsize=10)
    ax4.set_title('mAP Comparison: @K[1,5,10,15,20] vs @K[1,max]', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Precision @ K
    ax5 = plt.subplot(2, 4, 5)
    for result in results:
        precisions = [result[f'P@{k}'] for k in k_values]
        linestyle = '-' if 'Euclidean' in result['name'] else '--'
        marker = 'o' if not result['pca'] else 's'
        ax5.plot(k_values, precisions, marker=marker, linestyle=linestyle, 
                label=result['name'], linewidth=2, markersize=6)
    ax5.set_xlabel('K', fontsize=10)
    ax5.set_ylabel('Precision', fontsize=10)
    ax5.set_title('Precision @ K', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=6, loc='best', ncol=2)
    ax5.grid(alpha=0.3)
    
    # 6. Recall @ K
    ax6 = plt.subplot(2, 4, 6)
    for result in results:
        recalls = [result[f'R@{k}'] for k in k_values]
        linestyle = '-' if 'Euclidean' in result['name'] else '--'
        marker = 'o' if not result['pca'] else 's'
        ax6.plot(k_values, recalls, marker=marker, linestyle=linestyle,
                label=result['name'], linewidth=2, markersize=6)
    ax6.set_xlabel('K', fontsize=10)
    ax6.set_ylabel('Recall', fontsize=10)
    ax6.set_title('Recall @ K', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=6, loc='best', ncol=2)
    ax6.grid(alpha=0.3)
    
    # 7. P@10 vs Dimensions (efficiency)
    ax7 = plt.subplot(2, 4, 7)
    p_at_10 = [r['P@10'] for r in results]
    for i, result in enumerate(results):
        color = 'blue' if 'Euclidean' in result['name'] else 'red'
        marker = 'o' if not result['pca'] else 's'
        ax7.scatter(dims[i], p_at_10[i], s=150, c=color, marker=marker, 
                   alpha=0.7, edgecolors='black', linewidth=1.5)
        ax7.annotate(result['name'].split(' ')[0], (dims[i], p_at_10[i]), 
                    fontsize=6, ha='center', va='bottom')
    ax7.set_xlabel('Dimensions', fontsize=10)
    ax7.set_ylabel('Precision @ 10', fontsize=10)
    ax7.set_title('Efficiency: P@10 vs Dims', fontsize=11, fontweight='bold')
    ax7.grid(alpha=0.3)
    
    # 8. Improvement over Baselines (using mAP@K[1,max])
    ax8 = plt.subplot(2, 4, 8)
    baseline_euc = next(r for r in results if 'Baseline (Euclidean)' in r['name'])
    baseline_mah = next(r for r in results if 'Baseline (Mahalanobis)' in r['name'])
    
    baseline_map_euc = baseline_euc['best_map_all']
    baseline_map_mah = baseline_mah['best_map_all']
    
    group1_results = results[:len(EXPERIMENTS)//2]
    group2_results = results[len(EXPERIMENTS)//2:]
    group1_names = names[:len(EXPERIMENTS)//2]
    group2_names = names[len(EXPERIMENTS)//2:]
    
    improvements_g1 = [(r['best_map_all'] - baseline_map_euc) / baseline_map_euc * 100 for r in group1_results]
    improvements_g2 = [(r['best_map_all'] - baseline_map_mah) / baseline_map_mah * 100 for r in group2_results]
    
    y1 = np.arange(len(group1_names))
    y2 = np.arange(len(group2_names)) + len(group1_names) + 1
    
    ax8.barh(y1, improvements_g1, color='skyblue', alpha=0.7, label='vs Baseline (Euc)')
    ax8.barh(y2, improvements_g2, color='lightcoral', alpha=0.7, label='vs Baseline (Mah)')
    ax8.set_yticks(np.concatenate([y1, y2]))
    ax8.set_yticklabels(group1_names + group2_names, fontsize=8)
    ax8.set_xlabel('nAP@K[1,max] Improvement (%)', fontsize=10)
    ax8.set_title('Improvement (mAP@K[1,max])', fontsize=11, fontweight='bold')
    ax8.axvline(x=0, color='black', linewidth=1)
    ax8.axhline(y=len(group1_names) - 0.5, color='gray', linestyle='--', alpha=0.6)
    ax8.grid(axis='x', alpha=0.3)
    ax8.legend(fontsize=8, loc='lower right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = f'{descriptor_subfolder}_pca_mahalanobis_comparison.png'
    save_path = get_export_path(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()

def save_results_table(results, descriptor_subfolder):
    """Save results to CSV with both mAP metrics"""
    
    df = pd.DataFrame(results)
    
    col_order = ['name', 'pca', 'distance', 'dimensions', 'map_at_k', 'best_map_all'] + \
                [col for col in df.columns if col.startswith('P@') or col.startswith('R@')]
    df = df[col_order]
    
    csv_path = f'{descriptor_subfolder}_pca_comparison_results.csv'
    csv_path = get_export_path(csv_path)
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nResults saved to: {csv_path}")
    
    print("\n" + "="*120)
    print("RESULTS SUMMARY")
    print("="*120)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print("="*120)
    
    # Find best using mAP@K[1,max]
    best_idx = df['best_map_all'].idxmax()
    best_config = df.iloc[best_idx]
    
    print(f"\nBEST CONFIGURATION (by mAP):")
    print(f"   Method: {best_config['name']}")
    print(f"   mAP@K[1,5,10,15,20]: {best_config['map_at_k']:.4f}")
    print(f"   mAP@K[1,max]: {best_config['best_map_all']:.4f}")
    print(f"   Dimensions: {int(best_config['dimensions'])}")
    print(f"   P@10: {best_config['P@10']:.4f}")
    
    baseline_map = df.iloc[0]['best_map_all']
    improvement = (best_config['best_map_all'] - baseline_map) / baseline_map * 100
    print(f"   Improvement over baseline: {improvement:.2f}%")

def save_combined_summary(all_descriptor_results):
    """Save summary with both mAP metrics"""
    
    summary_data = []
    for descriptor, results in all_descriptor_results.items():
        baseline_euc = next(r for r in results if 'Baseline (Euclidean)' in r['name'])
        baseline_mah = next(r for r in results if 'Baseline (Mahalanobis)' in r['name'])
        best_overall = max(results, key=lambda r: r['best_map_all'])
        
        summary_data.append({
            'descriptor': descriptor,
            'baseline_euc_map_k': baseline_euc['map_at_k'],
            'baseline_euc_best_map_all': baseline_euc['best_map_all'],
            'baseline_mah_map_k': baseline_mah['map_at_k'],
            'baseline_mah_best_map_all': baseline_mah['best_map_all'],
            'best_method': best_overall['name'],
            'best_map_k': best_overall['map_at_k'],
            'best_map_all': best_overall['best_map_all'],
            'best_dims': best_overall['dimensions'],
            'improvement_vs_euc_%': (best_overall['best_map_all'] - baseline_euc['best_map_all']) / baseline_euc['best_map_all'] * 100
        })
    
    df_summary = pd.DataFrame(summary_data)
    csv_path = 'all_descriptors_pca_comparison.csv'
    csv_path = get_export_path(csv_path)
    df_summary.to_csv(csv_path, index=False, float_format='%.4f')
    
    print("\n" + "="*150)
    print("SUMMARY ACROSS ALL DESCRIPTORS")
    print("="*150)
    print(df_summary.to_string(index=False, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))
    print("="*150)
    print(f"\nCombined summary saved to: {csv_path}")

def get_export_path(filename):
    """
    Automatically places output files into the right subfolders under 'pca_exports/'.
    Handles spatial descriptors, global descriptors, and summary files.
    """
    # --- If path already contains directory structure, ensure it exists and return ---
    if filename.startswith("pca_exports") or os.path.sep in filename:
        # Extract directory and ensure it exists
        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)
        return filename

    # --- combined summary ---
    if filename.startswith("pca_comparison_SUMMARY_ALL_DESCRIPTORS"):
        folder = os.path.join("pca_exports")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)

    # --- spatialGrid_X descriptors ---
    match_desc = re.match(r"^(spatialGrid_[^_]+_[^_]+_[^_]+_[^_]+)", filename)
    if match_desc:
        descriptor = match_desc.group(1)
        folder = os.path.join("pca_exports", descriptor)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)

    # --- global descriptors like 'globalRGBhisto_color_4bins' ---
    match_global = re.match(r"^(global[^_]+)", filename)
    if match_global:
        descriptor = match_global.group(1)
        folder = os.path.join("pca_exports", descriptor)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)

    # --- default ---
    folder = "pca_exports"
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

if __name__ == "__main__":

    # Configuration
    DESCRIPTOR_FOLDER = 'descriptors'
    DESCRIPTOR_SUBFOLDERS = [
        # # Color descriptors with NO GRID (grid=1 + color bins)
            # 'globalRGBhisto_color_4bins',
            # 'globalRGBhisto_color_8bins',
            # 'globalRGBhisto_color_12bins',
            # 'globalRGBhisto_color_16bins',
            # 'globalRGBhisto_color_20bins',
            # 'globalRGBhisto_color_24bins',
            # 'globalRGBhisto_color_28bins',
            # 'globalRGBhisto_color_32bins',

            # # Color descriptors (all grids × color bins)
            # 'spatialGrid_2x2_color_4bins',
            # 'spatialGrid_3x3_color_4bins',
            # 'spatialGrid_4x4_color_4bins',

            # 'spatialGrid_2x2_color_8bins',
            # 'spatialGrid_3x3_color_8bins',
            # 'spatialGrid_4x4_color_8bins',

            # 'spatialGrid_2x2_color_12bins',
            # 'spatialGrid_3x3_color_12bins',
            # 'spatialGrid_4x4_color_12bins',

            # 'spatialGrid_2x2_color_16bins',
            # 'spatialGrid_3x3_color_16bins',
            # 'spatialGrid_4x4_color_16bins',


            # # Texture descriptors (all grids × orientation bins)
            # 'spatialGrid_2x2_texture_4orient',
            # 'spatialGrid_3x3_texture_4orient',
            # 'spatialGrid_4x4_texture_4orient',

            # 'spatialGrid_2x2_texture_8orient',
            # 'spatialGrid_3x3_texture_8orient',
            # 'spatialGrid_4x4_texture_8orient',

            # 'spatialGrid_2x2_texture_12orient',
            # 'spatialGrid_3x3_texture_12orient',
            # 'spatialGrid_4x4_texture_12orient',

            # 'spatialGrid_2x2_texture_16orient',
            # 'spatialGrid_3x3_texture_16orient',
            # 'spatialGrid_4x4_texture_16orient',


            # Combined color + texture (various bins & grids)
            # 'spatialGrid_2x2_both_4bins_4orient',
            # 'spatialGrid_3x3_both_4bins_4orient',
            # 'spatialGrid_4x4_both_4bins_4orient',

            'spatialGrid_2x2_both_8bins_4orient',
            'spatialGrid_3x3_both_8bins_4orient',
            'spatialGrid_4x4_both_8bins_4orient',

            # 'spatialGrid_2x2_both_4bins_8orient',
            # 'spatialGrid_3x3_both_4bins_8orient',
            # 'spatialGrid_4x4_both_4bins_8orient',

            'spatialGrid_2x2_both_8bins_8orient',
            'spatialGrid_3x3_both_8bins_8orient',
            'spatialGrid_4x4_both_8bins_8orient',

            # 'spatialGrid_2x2_both_12bins_12orient',
            # 'spatialGrid_3x3_both_12bins_12orient',
            # 'spatialGrid_4x4_both_12bins_12orient',

            'spatialGrid_2x2_both_16bins_16orient',
            'spatialGrid_3x3_both_16bins_16orient',
            'spatialGrid_4x4_both_16bins_16orient',
    ]

    IMAGES_FOLDER = 'msrc_objcategimagedatabase_v2/Images'
    PCA_DESCRIPTOR_FOLDER = 'pca_descriptors'

    # Experimental configurations to test
    EXPERIMENTS = [
        {'name': 'Baseline (Euclidean)', 'pca': False, 'distance': 'euclidean'},
        {'name': 'PCA 70% (Euclidean)', 'pca': True, 'variance': 0.70, 'distance': 'euclidean'},
        {'name': 'PCA 75% (Euclidean)', 'pca': True, 'variance': 0.75, 'distance': 'euclidean'},
        {'name': 'PCA 80% (Euclidean)', 'pca': True, 'variance': 0.80, 'distance': 'euclidean'},
        {'name': 'PCA 85% (Euclidean)', 'pca': True, 'variance': 0.85, 'distance': 'euclidean'},
        {'name': 'PCA 90% (Euclidean)', 'pca': True, 'variance': 0.90, 'distance': 'euclidean'},
        {'name': 'PCA 95% (Euclidean)', 'pca': True, 'variance': 0.95, 'distance': 'euclidean'},
        {'name': 'PCA 99% (Euclidean)', 'pca': True, 'variance': 0.99, 'distance': 'euclidean'},
        {'name': 'Baseline (Mahalanobis)', 'pca': False, 'distance': 'mahalanobis'},
        {'name': 'PCA 70% (Mahalanobis)', 'pca': True, 'variance': 0.70, 'distance': 'mahalanobis'},
        {'name': 'PCA 75% (Mahalanobis)', 'pca': True, 'variance': 0.75, 'distance': 'mahalanobis'},
        {'name': 'PCA 80% (Mahalanobis)', 'pca': True, 'variance': 0.80, 'distance': 'mahalanobis'},
        {'name': 'PCA 85% (Mahalanobis)', 'pca': True, 'variance': 0.85, 'distance': 'mahalanobis'},
        {'name': 'PCA 90% (Mahalanobis)', 'pca': True, 'variance': 0.90, 'distance': 'mahalanobis'},
        {'name': 'PCA 95% (Mahalanobis)', 'pca': True, 'variance': 0.95, 'distance': 'mahalanobis'},
        {'name': 'PCA 99% (Mahalanobis)', 'pca': True, 'variance': 0.99, 'distance': 'mahalanobis'},
    ]

    all_descriptor_results = {}
    
    for idx, descriptor_subfolder in enumerate(DESCRIPTOR_SUBFOLDERS, 1):
        print("\n\n" + "#"*80)
        print(f"# PROCESSING DESCRIPTOR {idx}/{len(DESCRIPTOR_SUBFOLDERS)}: {descriptor_subfolder}")
        print("#"*80 + "\n")
        
        try:
            results, k_values = run_all_experiments(descriptor_subfolder)
            all_descriptor_results[descriptor_subfolder] = results
            
            print("\n" + "="*70)
            print("GENERATING COMPARISON PLOTS")
            print("="*70)
            plot_comparison_results(results, k_values, descriptor_subfolder)
            save_results_table(results, descriptor_subfolder)
            
        except Exception as e:
            print(f"\n⚠ ERROR processing {descriptor_subfolder}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if all_descriptor_results:
        save_combined_summary(all_descriptor_results)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"\nProcessed {len(all_descriptor_results)} descriptors successfully")
    print("\nGenerated files:")
    print("  - *_pca_mahalanobis_comparison.png (one per descriptor)")
    print("  - *_pca_comparison_results.csv (one per descriptor)")
    print("  - all_descriptors_pca_comparison.csv (combined summary)")