# ------------------------------------------
# | Full Name: Lampros Grammatikopoulos    |
# | University email: lg01302@surrey.ac.uk |
# | University number: 6918674             |
# ------------------------------------------

import os
import sys
import re
sys.path.append('..')  # Add parent directory to Python path
import numpy as np
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from cvpr_compare import cvpr_compare
from cvpr_pca_utils import compute_covariance_matrix

def get_category_from_filename(filename):
    """Extract category from MSRC filename (e.g., '1_2_s.bmp' -> '1')"""
    basename = os.path.splitext(os.path.basename(filename))[0]
    return basename.split('_')[0]

def load_all_descriptors(descriptor_subfolder):
    """Load all descriptors and build lookup structures"""
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
                    descriptor = img_data['Hist'].flatten()  # Ensure 1D
                    ALLFEAT.append(descriptor)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    ALLFEAT = np.array(ALLFEAT)
    print(f"Loaded {len(ALLFEAT)} descriptors successfully.")
    
    return ALLFEAT, ALLFILES

def print_combined_statistics(all_results):
    """Print statistics for all descriptors"""
    print("\n" + "="*70)
    print("COMBINED EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n{'Descriptor':<40} {'MAP Score':<15}")
    print("-" * 60)
    for descriptor_name, _, _, map_score in all_results:
        print(f"{descriptor_name:<40} {map_score:<15.4f}")
    
    print("\n" + "="*70)
    
def retrieve_top_k(query_idx, ALLFEAT, k=10, metric='euclidean', inv_cov=None, p=0):
    """Retrieve top-k most similar images to query"""
    query = ALLFEAT[query_idx]
    distances = [(cvpr_compare(query, ALLFEAT[i], metric=metric, inv_cov=inv_cov, p=p), i) for i in range(len(ALLFEAT)) if i != query_idx]
    distances.sort(key=lambda x: x[0])
    return distances[:k]

def compute_precision_recall_single(query_idx, retrieved_results, ALLFILES, k_values):
    """Compute precision and recall at specific k values"""
    query_category = get_category_from_filename(os.path.basename(ALLFILES[query_idx]))
    total_relevant = sum(1 for f in ALLFILES 
                        if get_category_from_filename(os.path.basename(f)) == query_category) - 1
    pr_stats = {}
    for k in k_values:
        if k == 0:
            pr_stats[k] = {'precision':0.0, 'recall':0.0, 'relevant_retrieved':0}
            continue
        top_k = retrieved_results[:min(k, len(retrieved_results))]
        relevant_retrieved = sum(1 for _, idx in top_k 
                                if get_category_from_filename(os.path.basename(ALLFILES[idx])) == query_category)
        precision = relevant_retrieved / k
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
        pr_stats[k] = {'precision':precision, 'recall':recall, 'relevant_retrieved':relevant_retrieved}
    return pr_stats, query_category

def compute_full_pr_curve_single(query_idx, ALLFEAT, ALLFILES, metric='euclidean', inv_cov=None, p=0, full_dataset=False):
    """
    Compute full precision-recall curve
    
    Key fixes:
    1. Only record precision/recall when finding relevant items (not at every position)
    2. Stop after finding all relevant items (unless full_dataset=True)
    3. Add starting point
    """
    query_category = get_category_from_filename(os.path.basename(ALLFILES[query_idx]))

    # Compute distances to ALL other images
    distances = []
    for i in range(len(ALLFEAT)):
        if i != query_idx:
            dist = cvpr_compare(ALLFEAT[query_idx], ALLFEAT[i], metric=metric, inv_cov=inv_cov, p=p)
            distances.append((dist, i))
    
    distances.sort(key=lambda x: x[0])
    
    # Count total relevant images
    total_relevant = sum(1 for f in ALLFILES 
                         if get_category_from_filename(os.path.basename(f)) == query_category) - 1
    
    if total_relevant <= 0:
        return [0.0, 1.0], [0.0, 0.0]

    # Record precision/recall ONLY when finding relevant items
    precisions, recalls = [], []
    relevant_so_far = 0
    
    for k, (_, idx) in enumerate(distances, start=1):
        is_relevant = get_category_from_filename(os.path.basename(ALLFILES[idx])) == query_category
        
        if is_relevant:
            relevant_so_far += 1
            # Only record when we find a relevant item
            precision = relevant_so_far / k
            recall = relevant_so_far / total_relevant
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Stop when we've found all relevant items (unless full_dataset=True)
        if not full_dataset and relevant_so_far == total_relevant:
            break

    # Handle case where no relevant items found
    if len(precisions) == 0:
        return [0.0, 1.0], [0.0, 0.0]

    # Add starting point (recall=0, precision=1.0 or first precision)
    # Standard practice: precision=1.0 at recall=0
    recalls = [0.0] + recalls
    precisions = [1.0] + precisions
    
    return recalls, precisions

def evaluate_all_queries(ALLFEAT, ALLFILES, k_values=[0,1,2,3,4,5,6,7,8,9,10], full_curve=False, metric='euclidean', inv_cov=None, p=0):
    """
    Evaluate all queries for either:
    - full PR curve with MAP (full_curve=True)
    - top-k precision/recall (full_curve=False)
    """
    if full_curve:
        # --- Full PR curve + MAP calculation ---
        all_recalls_full, all_precisions_full, all_ap_scores = [], [], []

        for idx in range(len(ALLFEAT)):
            if idx % 50 == 0:
                print(f"Processing query {idx}/{len(ALLFEAT)} (full PR curve)...")
            
            # Use full_dataset=False to stop after finding all relevant items
            recalls, precisions = compute_full_pr_curve_single(
                idx, ALLFEAT, ALLFILES, metric=metric, inv_cov=inv_cov, p=p, 
                full_dataset=False  # Don't go through entire dataset
            )
            
            # Calculate AP: Average of precisions at recall points where relevant items are found
            if len(precisions) > 1:
                # Remove the starting point (recall=0) before averaging
                ap_score = np.mean(precisions[1:])  # Simple average, not trapezoid
            else:
                ap_score = 0.0
            
            all_recalls_full.append(recalls)
            all_precisions_full.append(precisions)
            all_ap_scores.append(ap_score)

        # Interpolate all curves to common recall levels for plotting
        recall_levels = np.linspace(0, 1, 101)
        interpolated_precisions = []
        for r, p in zip(all_recalls_full, all_precisions_full):
            if len(r) > 0 and len(p) > 0:
                interp_p = np.interp(recall_levels, r, p, left=p[0], right=p[-1])
            else:
                interp_p = np.zeros(len(recall_levels))
            interpolated_precisions.append(interp_p)
        
        avg_precision_curve = np.mean(interpolated_precisions, axis=0)
        map_score = np.mean(all_ap_scores)  # MAP: mean of per-query AP

        return {'recall_levels': recall_levels, 'precisions': avg_precision_curve, 'map_score': map_score, 'type':'full'}
    
    else:
        # --- Top-k evaluation ---
        all_precisions, all_recalls, confusion_data = {k:[] for k in k_values}, {k:[] for k in k_values}, []

        print(f"\nEvaluating all {len(ALLFEAT)} queries (top-k)...")
        for idx in range(len(ALLFEAT)):
            if idx % 50 == 0: print(f"Processing query {idx}/{len(ALLFEAT)} (top-k)...")
            retrieved = retrieve_top_k(idx, ALLFEAT, k=max(k_values), metric=metric, inv_cov=inv_cov, p=p)
            pr_stats, query_category = compute_precision_recall_single(idx, retrieved, ALLFILES, k_values)
            
            for k in k_values:
                all_precisions[k].append(pr_stats[k]['precision'])
                all_recalls[k].append(pr_stats[k]['recall'])
            
            if retrieved:
                top1_idx = retrieved[0][1]
                predicted_category = get_category_from_filename(os.path.basename(ALLFILES[top1_idx]))
                confusion_data.append((query_category, predicted_category))

        avg_precision = {k: np.mean(all_precisions[k]) for k in k_values}
        avg_recall = {k: np.mean(all_recalls[k]) for k in k_values}

        map_from_k = np.mean([avg_precision[k] for k in k_values if k > 0])
        return avg_precision, avg_recall, confusion_data

def evaluate_pr_per_class_all_queries(ALLFEAT, ALLFILES, metric='euclidean', inv_cov=None, p=0):
    """Compute full P-R curves and MAP per class"""

    categories = sorted({get_category_from_filename(os.path.basename(f)) for f in ALLFILES})
    class_results = {}

    for cat in categories:
        query_indices = [i for i, f in enumerate(ALLFILES) if get_category_from_filename(os.path.basename(f)) == cat]
        all_recalls_full, all_precisions_full = [], []

        for idx in query_indices:
            recalls, precisions = compute_full_pr_curve_single(idx, ALLFEAT, ALLFILES, metric=metric, inv_cov=inv_cov, p=p)
            all_recalls_full.append(recalls)
            all_precisions_full.append(precisions)

        # Interpolate all curves to common recall levels
        recall_levels = np.linspace(0, 1, 101)
        interpolated_precisions = []
        for rec, pres in zip(all_recalls_full, all_precisions_full):
            if len(rec) > 0 and len(pres) > 0:
                interp_pres = np.interp(recall_levels, rec, pres, left=pres[0], right=pres[-1])
            else:
                interp_pres = np.zeros(len(recall_levels))
            interpolated_precisions.append(interp_pres)

        mean_prec_full = np.mean(interpolated_precisions, axis=0)
        map_full = np.mean(mean_prec_full)

        class_results[cat] = {
            'recall_levels': recall_levels,
            'precisions_full': mean_prec_full,
            'map_full': map_full
        }

    return class_results

def plot_map_pr_curves_per_class(class_results, descriptor_name, save_path='per_class_pr_curves.png'):
    """
    Plot Precision-Recall curves for each class with MAP displayed and
    the overall mean P-R curve.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(14, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_results)))
    
    all_precisions = []

    # Plot each class curve
    for i, (cat, res) in enumerate(sorted(class_results.items())):
        plt.plot(res['recall_levels'], res['precisions_full'],
                 color=colors[i],
                 linewidth=1.8,
                 label=f'{cat} (MAP={res["map_full"]:.3f})')
        all_precisions.append(res['precisions_full'])

    # --- Compute and plot mean (macro-averaged) curve ---
    recall_levels = next(iter(class_results.values()))['recall_levels']
    mean_precision_all = np.mean(all_precisions, axis=0)
    mean_map_all = np.mean([res['map_full'] for res in class_results.values()])

    plt.plot(recall_levels, mean_precision_all, 'k--', linewidth=3, label=f'Mean (MAP={mean_map_all:.3f})')

    # --- Formatting ---
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title(f'Per-Class Precision–Recall Curves – {descriptor_name}',
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(fontsize=10, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    save_path = get_export_path(save_path)
    plt.savefig(save_path, dpi=300)
    print(f"Per-class + mean PR curves saved to {save_path}")
    plt.close()

def plot_confusion_matrix(confusion_data, descriptor_name, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    # Get unique categories
    all_categories = set()
    for true_cat, pred_cat in confusion_data:
        all_categories.add(true_cat)
        all_categories.add(pred_cat)
    
    categories = sorted(all_categories)
    n_categories = len(categories)
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    # Build confusion matrix
    confusion_matrix = np.zeros((n_categories, n_categories))
    for true_cat, pred_cat in confusion_data:
        true_idx = cat_to_idx[true_cat]
        pred_idx = cat_to_idx[pred_cat]
        confusion_matrix[true_idx, pred_idx] += 1
    
    # Normalize by row
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    confusion_matrix_norm = confusion_matrix / (row_sums + 1e-10)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix_norm, annot=True, fmt='.2f', 
                cmap='Blues', xticklabels=categories, yticklabels=categories,
                cbar_kws={'label': 'Proportion'})
    plt.xlabel('Predicted Category', fontsize=12)
    plt.ylabel('True Category', fontsize=12)
    plt.title(f'Confusion Matrix - {descriptor_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = get_export_path(save_path)
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {save_path}")
    # plt.show()
    plt.close()

def display_query_results(query_idx, retrieved, ALLFILES, descriptor_name, distance_metric, show=10):
    """Display query image and top matches"""
    images_to_show = []
    titles = []
    
    # Add query image first
    query_file = os.path.basename(ALLFILES[query_idx])
    query_path = os.path.join(IMAGES_FOLDER, query_file.replace('.mat', '.bmp'))
    query_category = get_category_from_filename(query_file)
    
    query_img = cv2.imread(query_path)
    if query_img is not None:
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        query_img = cv2.resize(query_img, (150, 150))
        images_to_show.append(query_img)
        titles.append(f"QUERY\nCat: {query_category}")
    
    # Add retrieved images
    for i, (dist, idx) in enumerate(retrieved[:show]):
        result_file = os.path.basename(ALLFILES[idx])
        result_path = os.path.join(IMAGES_FOLDER, result_file.replace('.mat', '.bmp'))
        result_category = get_category_from_filename(result_file)
        
        img = cv2.imread(result_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (150, 150))
            images_to_show.append(img)
            
            # Color code: green if correct category, red if wrong
            color = 'green' if result_category == query_category else 'red'
            titles.append(f"#{i+1} (d={dist:.3f})\nCat: {result_category}")
    
    # Display in grid
    n_imgs = len(images_to_show)
    cols = min(6, n_imgs)
    rows = (n_imgs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.1*rows))
    axes = axes.flatten() if n_imgs > 1 else [axes]
    
    # Add descriptor as a single title at the top
    fig.suptitle(f"Descriptor: {descriptor_name}", fontsize=16)
    
    for i, (img, title) in enumerate(zip(images_to_show, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=9)
        axes[i].axis('off')
        
        # Add colored border for matches
        if i > 0:  # Skip query
            result_file = os.path.basename(ALLFILES[retrieved[i-1][1]])
            result_category = get_category_from_filename(result_file)
            color = 'green' if result_category == query_category else 'red'
            for spine in axes[i].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
    
    # Hide unused subplots
    for i in range(n_imgs, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = f'{descriptor_name}_{distance_metric}_query_results.png'
    save_path = get_export_path(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Query results saved to {descriptor_name}_{distance_metric}_query_results.png")
    # plt.show()
    plt.close(fig)

def print_statistics(avg_precision, avg_recall, descriptor_name):
    """Print formatted statistics"""
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - {descriptor_name}")
    print("="*60)
    
    k_values = sorted(avg_precision.keys())
    print(f"\n{'K':<8} {'Precision':<12} {'Recall':<12}")
    print("-" * 35)
    for k in k_values:
        print(f"{k:<8} {avg_precision[k]:<12.4f} {avg_recall[k]:<12.4f}")
    
    map_score = np.mean([avg_precision[k] for k in k_values if k > 0])
    print(f"\nMAP (Mean Average Precision): {map_score:.4f}")

def plot_combined_pr_curves(all_results, all_k_results, save_path_full='combined_pr_curves_full.png', save_path_top10='combined_pr_curves_top10.png'):
    """
    Plot all PR curves - full curves and actual top-10 k values.
    """

    # Organize by feature type
    color_results = [r for r in all_results if 'color' in r[0] and 'both' not in r[0]]
    texture_results = [r for r in all_results if 'texture' in r[0] and 'both' not in r[0]]
    both_results = [r for r in all_results if 'both' in r[0]]
    
    # Color palette
    colors_color = plt.cm.Blues(np.linspace(0.4, 0.9, max(1, len(color_results))))
    colors_texture = plt.cm.Oranges(np.linspace(0.4, 0.9, max(1, len(texture_results))))
    colors_both = plt.cm.Greens(np.linspace(0.4, 0.9, max(1, len(both_results))))
    
    all_colors = list(colors_color)[:len(color_results)] + list(colors_texture)[:len(texture_results)] + list(colors_both)[:len(both_results)]
    markers = ['o', 's', '^', 'D', '*', 'v', 'P', 'X', 'H']  # extend if needed
    
    # Full Curves
    plt.figure(figsize=(14, 10))
    for i, (descriptor_name, recall_levels, precisions, map_score) in enumerate(all_results):
        label = descriptor_name.replace('spatialGrid_', '')
        plt.plot(recall_levels, precisions, 
                 linewidth=2, 
                 color=all_colors[i],
                 label=f'{label} (MAP={map_score:.3f})',
                 alpha=0.7)
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title('Precision-Recall Curves Comparison (Full)', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(fontsize=9, loc='upper right', framealpha=0.95, ncol=2)
    plt.tight_layout()
    save_path_full = get_export_path(save_path_full)
    plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
    print(f"Combined full PR curves saved to {save_path_full}")
    # plt.show()
    plt.close()
    
    # Top-10 Points (actual k=1,2,3...10 results)
    plt.figure(figsize=(14, 10))
    for i, (descriptor_name, avg_precision, avg_recall, map_from_k) in enumerate(all_k_results):
        label = descriptor_name.replace('spatialGrid_', '')
        
        # Extract k=1 through k=10 values
        k_values = sorted([k for k in avg_precision.keys() if 1 <= k <= 10])
        precisions_at_k = [avg_precision[k] for k in k_values]
        recalls_at_k = [avg_recall[k] for k in k_values]
        
        # Add starting point at (0, precision@k=1) to connect to y-axis
        # This is standard: at recall=0, use the first precision value
        recalls_at_k = [0.0] + recalls_at_k
        precisions_at_k = [precisions_at_k[0]] + precisions_at_k
        
        # Plot as continuous line with markers
        plt.plot(recalls_at_k, precisions_at_k,
                 linewidth=2.5, 
                 color=all_colors[i],
                 label=f'{label} (MAP@k={map_from_k:.3f})',
                 alpha=0.8,
                 marker=markers[i % len(markers)],
                 markersize=10,
                 markeredgewidth=1.5,
                 markeredgecolor='white')
    
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title('Precision-Recall at Top-K (k=1 to 10)', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0, max([max(recalls_at_k[1:]) for _, avg_prec, avg_rec, _ in all_k_results 
                      for recalls_at_k in [[avg_rec[k] for k in sorted(avg_rec.keys()) if 1 <= k <= 10]]]) * 1.1])
    # plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(fontsize=9, loc='upper right', framealpha=0.95, ncol=2)
    plt.tight_layout()
    save_path_top10 = get_export_path(save_path_top10)
    plt.savefig(save_path_top10, dpi=300, bbox_inches='tight')
    print(f"Combined top-10 PR points saved to {save_path_top10}")
    plt.close()

def plot_ap_one_query_pr_curves_per_class(ALLFEAT, ALLFILES, descriptor_name, metric='euclidean', save_path='per_class_pr_curves.png', inv_cov=None, p=0):
    """
    Plot PR curves for one sample query from each category.

    - Computes Average Precision (AP) for each sampled query (one per class).
    - Saves figure and prints per-category AP and P@1.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Map categories to their image indices
    category_to_indices = {}
    for idx, filepath in enumerate(ALLFILES):
        cat = get_category_from_filename(os.path.basename(filepath))
        category_to_indices.setdefault(cat, []).append(idx)
    categories = sorted(category_to_indices.keys())

    print(f"\nGenerating per-class PR curves for {len(categories)} categories...")

    # Prepare figure
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    plt.figure(figsize=(14, 10))

    # Compute AP and PR curves for one query per class
    for i, cat in enumerate(categories):
        query_idx = category_to_indices[cat][0]  # take first sample query
        recalls, precisions = compute_full_pr_curve_single(query_idx, ALLFEAT, ALLFILES, metric=metric, inv_cov=inv_cov, p=p)

        # Average Precision (AP) for this query
        ap_score = np.mean(precisions[1:]) if len(precisions) > 1 else 0.0
        n_in_category = len(category_to_indices[cat])

        plt.plot(recalls, precisions, 
                 linewidth=1.5,
                 color=colors[i],
                 alpha=0.7,
                 label=f'Cat {cat} (n={n_in_category}, AP={ap_score:.3f})')

    # Figure formatting
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title(f'Per-Class PR Curves - {descriptor_name}\n(One sample query per category)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(fontsize=8, loc='upper right', framealpha=0.95, ncol=2)
    plt.tight_layout()
    save_path = get_export_path(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class PR curves saved to {save_path}")
    plt.close()

def plot_pr_curve_for_query_with_params(query_idx, descriptors_with_params, metric='euclidean', save_path='one_query_for_one_class_pr_curve_comparison.png'):
    """
    Plot PR curves for a single query across all descriptors with proper parameter handling.
    
    Parameters:
    - query_idx: index of the query image
    - descriptors_with_params: list of tuples (descriptor_name, ALLFEAT, ALLFILES, inv_cov, p)
    - metric: distance metric to use (default 'euclidean')
    - save_path: path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(14, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(descriptors_with_params)))

    for i, (descriptor_name, ALLFEAT, ALLFILES, inv_cov, p) in enumerate(descriptors_with_params):
        # Compute full PR curve for this query with parameters
        recalls, precisions = compute_full_pr_curve_single(query_idx, ALLFEAT, ALLFILES, metric=metric, inv_cov=inv_cov, p=p)
        if len(precisions) > 1:
            ap_score = np.mean(precisions[1:])
        else:
            ap_score = 0.0

        plt.plot(recalls, precisions, 
                 linewidth=2.5,
                 color=colors[i],
                 label=f'{descriptor_name} (AP={ap_score:.3f})',
                 alpha=0.8)

    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title(f'PR Curve for Query #{query_idx}', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(fontsize=10, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    save_path = get_export_path(save_path)
    plt.savefig(save_path, dpi=300)
    print(f"PR curves for query {query_idx} saved to {save_path}")
    plt.close()

def save_all_results_to_csv(all_descriptor_results, save_path='evaluation_results.csv'):
    """Save all evaluation results to a single CSV file"""
    import csv
    
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Descriptor', 'K', 'Precision', 'Recall', 'MAP@K[1-10]', 'MAP@K[1-max]'])
        
        # Write results for each descriptor
        for descriptor_name, avg_precision, avg_recall, map_from_k, map_full, recall_levels, precisions in all_descriptor_results:
            k_values = sorted(avg_precision.keys())
            
            # Calculate MAP@K[1-10] (average of precision at k=1 to 10 only)
            k_1_to_10 = [k for k in k_values if 1 <= k <= 10]
            map_k_1_to_10 = np.mean([avg_precision[k] for k in k_1_to_10]) if k_1_to_10 else 0.0
            
            for i, k in enumerate(k_values):
                # Only write MAP values in the first row for each descriptor
                map_1_10_str = f"{map_k_1_to_10:.6f}" if i == 0 else ''
                map_full_str = f"{map_full:.6f}" if i == 0 else ''
                
                writer.writerow([
                    descriptor_name,
                    k,
                    f"{avg_precision[k]:.6f}",
                    f"{avg_recall[k]:.6f}",
                    map_1_10_str,
                    map_full_str
                ])
            # Add blank line between descriptors
            writer.writerow([])
        
        # Add combined summary at the end
        writer.writerow(['=== COMBINED SUMMARY ==='])
        writer.writerow(['Descriptor', 'MAP@K[1-10]', 'MAP@K[1-max]'])
        for descriptor_name, avg_precision, avg_recall, map_from_k, map_full, recall_levels, precisions in all_descriptor_results:
            k_1_to_10 = [k for k in sorted(avg_precision.keys()) if 1 <= k <= 10]
            map_k_1_to_10 = np.mean([avg_precision[k] for k in k_1_to_10]) if k_1_to_10 else 0.0
            writer.writerow([descriptor_name, f"{map_k_1_to_10:.6f}", f"{map_full:.6f}"])
    
    print(f"\nAll evaluation results saved to {save_path}")

def plot_distance_metric_comparison(all_distance_metrics_results, k_values=None, save_path='all_descriptors_all_metrics_comparison.png'):
    """
    Plots a comparison of all distance metrics across all descriptor experiments.
    Parameters:
        all_distance_metrics_results: dict
            Keys: distance metric names
            Values: list of tuples per descriptor: (descriptor_name, avg_precision, avg_recall, map_from_k, map_score, recall_levels, precisions)
        k_values: list or None
            List of k values used (for top-k MAP comparison)
        save_path: str
            Path to save the plot
    """
    metrics = list(all_distance_metrics_results.keys())
    descriptors = [name for name, *_ in all_distance_metrics_results[metrics[0]]]
    
    # Prepare data
    map_full = []
    map_topk = []
    for metric in metrics:
        results = all_distance_metrics_results[metric]
        map_full.append([item[4] for item in results])  # map_score is index 4
        map_topk.append([item[3] for item in results])  # map_from_k is index 3
    
    map_full = np.array(map_full)
    map_topk = np.array(map_topk)
    
    x = np.arange(len(descriptors))
    width = 0.08
    
    # Create a diverse color palette
    colors = []
    colormaps = ['tab20', 'tab20b', 'tab20c', 'Set3', 'Pastel1', 'Pastel2']
    for cmap_name in colormaps:
        cmap = plt.colormaps[cmap_name]
        num_colors = cmap.N if hasattr(cmap, 'N') else 20
        colors.extend([cmap(i) for i in range(num_colors)])
        if len(colors) >= len(metrics):
            break
    
    # Ensure we have enough colors (fallback to generating from hsv)
    if len(colors) < len(metrics):
        additional_colors = plt.cm.hsv(np.linspace(0, 0.9, len(metrics) - len(colors)))
        colors.extend(additional_colors)
    
    # Full MAP plot
    plt.figure(figsize=(15, 6))
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, map_full[i], width, label=metric, color=colors[i], edgecolor='black', linewidth=0.5)
    
    plt.xticks(x + width*len(metrics)/2, descriptors, rotation=45, ha='right')
    plt.ylabel('MAP (Full Curve)', fontsize=12, fontweight='bold')
    plt.title('Comparison of Distance Metrics - Full MAP', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best', framealpha=0.9)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    save_path = get_export_path(save_path)
    plt.savefig(save_path, dpi=300)
    print(f"Full MAP comparison saved to {save_path}")
    plt.close()
    
    # Top-k MAP plot
    if k_values is not None:
        plt.figure(figsize=(15, 6))
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, map_topk[i], width, label=metric, color=colors[i], edgecolor='black', linewidth=0.5)
        
        plt.xticks(x + width*len(metrics)/2, descriptors, rotation=45, ha='right')
        plt.ylabel('MAP (Top-K)', fontsize=12, fontweight='bold')
        plt.title('Comparison of Distance Metrics - Top-K MAP', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best', framealpha=0.9)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        save_path = save_path.replace('.png', '_topk.png')
        save_path = get_export_path(save_path)
        plt.savefig(save_path, dpi=300)
        print(f"Top-K MAP comparison saved to {save_path}")
        plt.close()
    
    # PR Curves - SEPARATE FILE per descriptor with all metrics (no grid)
    num_descriptors = len(descriptors)
    
    for desc_idx, descriptor_name in enumerate(descriptors):
        # Create a new figure for each descriptor
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot PR curve for each metric
        for metric_idx, metric in enumerate(metrics):
            results = all_distance_metrics_results[metric]
            result_tuple = results[desc_idx]
            
            # Extract: (descriptor_name, avg_precision, avg_recall, map_from_k, map_score, recall_levels, precisions)
            if len(result_tuple) >= 7:
                descriptor_name_stored, avg_precision, avg_recall, map_from_k, map_score, recall_curve, precision_curve = result_tuple
                
                ax.plot(recall_curve, precision_curve, 
                       label=f'{metric} (MAP={map_score:.3f})',
                       color=colors[metric_idx], 
                       linewidth=2.5,
                       alpha=0.8)
            else:
                print(f"Warning: No PR curve data for {descriptor_name} with {metric}")
        
        ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
        ax.set_title(f'{descriptor_name}', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save with descriptor name in filename
        pr_curve_path = f'{descriptor_name}_all_metrics_pr_curve.png'
        pr_curve_path = get_export_path(pr_curve_path)
        plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
        print(f"PR curve for {descriptor_name} saved to {pr_curve_path}")
        plt.close()

def get_export_path(filename):
    """
    Automatically places output files into the right subfolders under 'visual_search_exports/'.
    Structure: 
    - visual_search_exports/all/ (cross-metric comparisons)
    - visual_search_exports/distance_metrics/METRIC/ (per-metric cross-descriptor)
    - visual_search_exports/DESCRIPTOR/METRIC/ (individual descriptor per metric)
    - visual_search_exports/DESCRIPTOR/ (descriptor all_metrics files)
    """
    # --- If path already contains directory structure, ensure it exists and return ---
    if filename.startswith("visual_search_exports") or os.path.sep in filename:
        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)
        return filename

    # --- Pattern 1: Cross-metric comparison files ---
    # all_descriptors_all_metrics_comparison.png
    # all_descriptors_all_metrics_comparison_topk.png
    if filename.startswith("all_descriptors_all_metrics_comparison"):
        folder = os.path.join("visual_search_exports", "all")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)

    # --- Pattern 2: Per-metric cross-descriptor files ---
    # all_descriptors_euclidean_mAP_PRs_full.png
    # all_descriptors_euclidean_mAP_PRs_top10.png
    # all_descriptors_euclidean_one_query.png
    # all_descriptors_euclidean_evaluation_results.csv
    if filename.startswith("all_descriptors_"):
        match_all_desc_metric = re.match(r"^all_descriptors_([a-z\-]+)_(.+)$", filename)
        if match_all_desc_metric:
            metric = match_all_desc_metric.group(1)
            rest = match_all_desc_metric.group(2)
            folder = os.path.join("visual_search_exports", "distance_metrics", metric)
            os.makedirs(folder, exist_ok=True)
            return os.path.join(folder, f"all_descriptors_{rest}")

    # --- Pattern 3: Descriptor with all_metrics files ---
    # spatialGrid_2x2_texture_4orient_all_metrics_pr_curve.png
    # globalRGBhisto_color_8bins_all_metrics_pr_curve.png
    # Clean filename by removing descriptor prefix
    if "_all_metrics_" in filename:
        match_desc_all_metrics = re.match(
            r"^(spatialGrid_\d+x\d+_color_\d+bins|spatialGrid_\d+x\d+_texture_\d+orient|spatialGrid_\d+x\d+_both_\d+bins_\d+orient|globalRGBhisto_color_\d+bins)_(all_metrics_.+)$",
            filename
        )
        if match_desc_all_metrics:
            descriptor = match_desc_all_metrics.group(1)
            cleaned_filename = match_desc_all_metrics.group(2)  # Just "all_metrics_pr_curve.png"
            folder = os.path.join("visual_search_exports", descriptor)
            os.makedirs(folder, exist_ok=True)
            return os.path.join(folder, cleaned_filename)

    # --- Pattern 4: Individual descriptor files WITH specific metric ---
    # spatialGrid_2x2_color_4bins_euclidean_confusion_matrix.png
    # spatialGrid_2x2_both_4bins_4orient_euclidean_AP_PRCs.png
    # spatialGrid_2x2_texture_4orient_euclidean_AP_PRCs.png
    # globalRGBhisto_color_8bins_euclidean_query_results.png
    match_desc_metric = re.match(
        r"^(spatialGrid_\d+x\d+_color_\d+bins|spatialGrid_\d+x\d+_texture_\d+orient|spatialGrid_\d+x\d+_both_\d+bins_\d+orient|globalRGBhisto_color_\d+bins)_([a-z\-]+)_(.+)$",
        filename
    )
    if match_desc_metric:
        descriptor = match_desc_metric.group(1)
        metric = match_desc_metric.group(2)
        rest_of_filename = match_desc_metric.group(3)
        
        # Create structure: descriptor/metric/cleaned_filename
        folder = os.path.join("visual_search_exports", descriptor, metric)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, rest_of_filename)

    # --- Pattern 5: Standalone descriptor files (no metric) ---
    # This shouldn't happen in your current code, but keeping for completeness
    match_desc = re.match(
        r"^(spatialGrid_\d+x\d+_color_\d+bins|spatialGrid_\d+x\d+_texture_\d+orient|spatialGrid_\d+x\d+_both_\d+bins_\d+orient|globalRGBhisto_color_\d+bins)$",
        filename
    )
    if match_desc:
        descriptor = match_desc.group(1)
        folder = os.path.join("visual_search_exports", descriptor)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)

    # --- Default: root visual_search_exports folder ---
    folder = "visual_search_exports"
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    """
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
    """
    
    # List of all distance metrics to test
    DISTANCE_METRICS = [
        'euclidean',
        'manhattan',
        'chebyshev',
        'cosine',
        'canberra',
        'mahalanobis',
        'minkowski',
        'chi-square',
        'intersection',
        'bhattacharyya',
        'hellinger',
        'correlation',
    ]

    # === Paths ===
    IMAGES_FOLDER = 'msrc_objcategimagedatabase_v2/Images'
    DESCRIPTOR_FOLDER = 'descriptors'
    DESCRIPTOR_SUBFOLDERS = [
        # Color descriptors with NO GRID (grid=1 + color bins)
        # 'globalRGBhisto_color_4bins',
        'globalRGBhisto_color_8bins',
        'globalRGBhisto_color_12bins',
        # 'globalRGBhisto_color_16bins',
        # 'globalRGBhisto_color_20bins',
        # 'globalRGBhisto_color_24bins',
        # 'globalRGBhisto_color_28bins',
        # 'globalRGBhisto_color_32bins',

        # Color descriptors (all grids × color bins)
        # 'spatialGrid_2x2_color_4bins',
        'spatialGrid_3x3_color_4bins',
        'spatialGrid_4x4_color_4bins',

        # 'spatialGrid_2x2_color_8bins',
        # 'spatialGrid_3x3_color_8bins',
        # 'spatialGrid_4x4_color_8bins',

        # 'spatialGrid_2x2_color_12bins',
        # 'spatialGrid_3x3_color_12bins',
        # 'spatialGrid_4x4_color_12bins',

        # 'spatialGrid_2x2_color_16bins',
        # 'spatialGrid_3x3_color_16bins',
        # 'spatialGrid_4x4_color_16bins',


        # Texture descriptors (all grids × orientation bins)
        # 'spatialGrid_2x2_texture_4orient',
        # 'spatialGrid_3x3_texture_4orient',
        # 'spatialGrid_4x4_texture_4orient',

        # 'spatialGrid_2x2_texture_8orient',
        # 'spatialGrid_3x3_texture_8orient',
        # 'spatialGrid_4x4_texture_8orient',

        # 'spatialGrid_2x2_texture_12orient',
        # 'spatialGrid_3x3_texture_12orient',
        # 'spatialGrid_4x4_texture_12orient',

        'spatialGrid_2x2_texture_16orient',
        'spatialGrid_3x3_texture_16orient',
        # 'spatialGrid_4x4_texture_16orient',


        # Combined color + texture (various bins & grids)
        # 'spatialGrid_2x2_both_4bins_4orient',
        # 'spatialGrid_3x3_both_4bins_4orient',
        # 'spatialGrid_4x4_both_4bins_4orient',

        # 'spatialGrid_2x2_both_8bins_4orient',
        # 'spatialGrid_3x3_both_8bins_4orient',
        # 'spatialGrid_4x4_both_8bins_4orient',

        # 'spatialGrid_2x2_both_4bins_8orient',
        'spatialGrid_3x3_both_4bins_8orient',
        'spatialGrid_4x4_both_4bins_8orient',

        # 'spatialGrid_2x2_both_8bins_8orient',
        # 'spatialGrid_3x3_both_8bins_8orient',
        # 'spatialGrid_4x4_both_8bins_8orient',

        # 'spatialGrid_2x2_both_12bins_12orient',
        # 'spatialGrid_3x3_both_12bins_12orient',
        # 'spatialGrid_4x4_both_12bins_12orient',

        # 'spatialGrid_2x2_both_16bins_16orient',
        # 'spatialGrid_3x3_both_16bins_16orient',
        # 'spatialGrid_4x4_both_16bins_16orient',
    ]

    k_values = list(range(11))
    all_distance_metrics_results = {}

    # Process each distance metric completely before moving to the next
    for DISTANCE_METRIC in DISTANCE_METRICS:
        # CLEAR all data structures for this metric
        all_results = []
        all_descriptor_results = []
        all_descriptors_data = []
        
        # Dictionary to store inv_cov and p for each descriptor
        descriptor_inv_covs = {}
        descriptor_p_values = {}
        
        print(f"\n{'='*70}")
        print(f"USING DISTANCE METRIC: {DISTANCE_METRIC.upper()}")
        print(f"{'='*70}\n")

        # Process each descriptor for this metric
        for descriptor_subfolder in DESCRIPTOR_SUBFOLDERS:
            print("\n" + "="*70)
            print(f"PROCESSING: {descriptor_subfolder}")
            print("="*70)

            # Load all descriptors (FRESH load for each descriptor)
            ALLFEAT_orig, ALLFILES = load_all_descriptors(descriptor_subfolder)

            if len(ALLFEAT_orig) == 0:
                print(f"Warning: No valid descriptors found for {descriptor_subfolder}!")
                continue

            # IMPORTANT: Reset parameters for each descriptor
            inv_cov = None
            p = 0

            # Apply PCA reduction ONLY for Mahalanobis distance with high-dimensional data
            if DISTANCE_METRIC == 'mahalanobis' and ALLFEAT_orig.shape[1] > 200:
                print(f"\n{'='*70}")
                print(f"Applying PCA for Mahalanobis distance")
                print(f"{'='*70}")
                from cvpr_pca_utils import compute_pca_projection
                
                original_dim = ALLFEAT_orig.shape[1]
                target_dim = min(100, ALLFEAT_orig.shape[0] // 2)
                
                pca_model, ALLFEAT, explained_var = compute_pca_projection(
                    ALLFEAT_orig, 
                    n_components=target_dim,
                    variance_threshold=0.95
                )
                
                print(f"PCA: {original_dim} → {ALLFEAT.shape[1]} dimensions")
                print(f"Explained variance: {explained_var:.2%}")
                print(f"{'='*70}\n")
            else:
                # Use original features for non-Mahalanobis metrics
                ALLFEAT = ALLFEAT_orig

            # Compute special parameters AFTER PCA (if applied)
            if DISTANCE_METRIC == 'mahalanobis':
                print("Computing covariance matrix for Mahalanobis distance...")
                cov_matrix, inv_cov = compute_covariance_matrix(ALLFEAT, regularization=1e-3)
                descriptor_inv_covs[descriptor_subfolder] = inv_cov
            elif DISTANCE_METRIC == 'minkowski':
                print("Setting parameter p for Minkowski distance...")
                p = int(3) # ← Make it explicitly an integer, not mutable!!!
                descriptor_p_values[descriptor_subfolder] = p

            # Store descriptors for cross-descriptor comparison (for this metric only)
            all_descriptors_data.append((descriptor_subfolder, ALLFEAT, ALLFILES))

            # Show random query
            print("\n" + "="*60)
            print("STEP 1: Query Retrieval")
            print("="*60)

            query_idx = np.random.randint(0, len(ALLFEAT))
            # query_idx = 355 # Image from class 2 - 'tree' class
            print(f"Query image index: {query_idx}")
            print(f"Query file: {os.path.basename(ALLFILES[query_idx])}")

            retrieved = retrieve_top_k(query_idx, ALLFEAT, k=10, metric=DISTANCE_METRIC, inv_cov=inv_cov, p=p)
            
            # Query results
            display_query_results(query_idx, retrieved, ALLFILES, descriptor_subfolder, DISTANCE_METRIC, show=10)

            # Full PR curve
            results_full = evaluate_all_queries(ALLFEAT, ALLFILES, full_curve=True, metric=DISTANCE_METRIC, inv_cov=inv_cov, p=p)

            # k-based PR stats
            results_k = evaluate_all_queries(ALLFEAT, ALLFILES, k_values=k_values, full_curve=False, metric=DISTANCE_METRIC, inv_cov=inv_cov, p=p)
            avg_precision, avg_recall, confusion_data = results_k

            # Print statistics
            print_statistics(avg_precision, avg_recall, descriptor_subfolder)

            # Calculate MAP from full curve
            map_score = results_full['map_score']
            print(f"MAP (from full curve): {map_score:.4f}")

            # Store results
            all_results.append((descriptor_subfolder, results_full['recall_levels'], results_full['precisions'], map_score))
            map_from_k = np.mean([avg_precision[k] for k in k_values if k > 0])
            # Store both the k-based stats AND full PR curve
            all_descriptor_results.append((
                descriptor_subfolder, 
                avg_precision,  # k-based precision dict
                avg_recall,      # k-based recall dict
                map_from_k,      # MAP from k values
                map_score,       # MAP from full curve
                results_full['recall_levels'],  # ADD: full recall array
                results_full['precisions']      # ADD: full precision array
            ))

            # Plot confusion matrix
            save_path = f'{descriptor_subfolder}_{DISTANCE_METRIC}_confusion_matrix.png'
            plot_confusion_matrix(confusion_data, descriptor_subfolder, save_path)

            # Class-level PR curves
            class_results = evaluate_pr_per_class_all_queries(ALLFEAT, ALLFILES, metric=DISTANCE_METRIC, inv_cov=inv_cov, p=p)
            save_path = f'{descriptor_subfolder}_{DISTANCE_METRIC}_mAP_PRCs.png'
            plot_map_pr_curves_per_class(class_results, descriptor_subfolder, save_path=save_path)

            save_path = f'{descriptor_subfolder}_{DISTANCE_METRIC}_AP_PRCs.png'
            plot_ap_one_query_pr_curves_per_class(ALLFEAT, ALLFILES, descriptor_subfolder, metric=DISTANCE_METRIC, save_path=save_path, inv_cov=inv_cov, p=p)

        # After processing all descriptors for this metric, save results
        all_distance_metrics_results[DISTANCE_METRIC] = all_descriptor_results

        # Create list with proper parameters for each descriptor
        descriptors_with_params = []
        for desc_name, ALLFEAT, ALLFILES in all_descriptors_data:
            desc_inv_cov = descriptor_inv_covs.get(desc_name, None)
            desc_p = descriptor_p_values.get(desc_name, 0)
            descriptors_with_params.append((desc_name, ALLFEAT, ALLFILES, desc_inv_cov, desc_p))
        
        # Plot PR curve comparison for ONE query across ALL descriptors (for this metric)
        save_path = f'all_descriptors_{DISTANCE_METRIC}_one_query.png'
        plot_pr_curve_for_query_with_params(query_idx, descriptors_with_params, metric=DISTANCE_METRIC, save_path=save_path)

        # Combined analysis for this metric
        print_combined_statistics(all_results)
        all_k_results = [(name, prec, rec, map_k) for name, prec, rec, map_k, map_score, recall_levels, precisions in all_descriptor_results]

        save_path_full = f"all_descriptors_{DISTANCE_METRIC}_mAP_PRs_full.png"
        save_path_top10 = f"all_descriptors_{DISTANCE_METRIC}_mAP_PRs_top10.png"
        plot_combined_pr_curves(all_results, all_k_results, save_path_full=save_path_full, save_path_top10=save_path_top10)
        save_path=f'all_descriptors_{DISTANCE_METRIC}_evaluation_results.csv'
        save_path = get_export_path(save_path)
        save_all_results_to_csv(all_descriptor_results, save_path=save_path)

        print("\n" + "="*70)
        print(f"ALL EVALUATIONS COMPLETE FOR {DISTANCE_METRIC.upper()}!")
        print("="*70)

    # After ALL metrics are done, compare them
    print("\n" + "="*70)
    print("COMPARING ALL DISTANCE METRICS")
    print("="*70)
    plot_distance_metric_comparison(all_distance_metrics_results, k_values=k_values, save_path='all_descriptors_all_metrics_comparison.png')