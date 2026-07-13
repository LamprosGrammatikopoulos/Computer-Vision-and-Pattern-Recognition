# ------------------------------------------
# | Full Name: Lampros Grammatikopoulos    |
# | University email: lg01302@surrey.ac.uk |
# | University number: 6918674             |
# ------------------------------------------

import os
import numpy as np
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
from descriptorFunctions import globalColorHistogramDescriptor, spatialGridHistogramDescriptor

DATASET_FOLDER = 'msrc_objcategimagedatabase_v2'
OUT_FOLDER = 'descriptors'
FEATURE_TYPE = 'both'

# Store all descriptors for final plotting
all_descriptors = {}

# ============================================= PART 1: GLOBAL HISTOGRAMS =============================================
print("\n" + "="*70)
print("PART 1: COMPUTING GLOBAL COLOR HISTOGRAMS")
print("="*70 + "\n")

QUANTIZATION_LEVELS = [4, 8, 12, 16, 20, 24, 28, 32] # Try 4x4x4=64, 8x8x8=512, 12x12x12=1728, 16x16x16=4096 bins...

for bins in QUANTIZATION_LEVELS:
    OUT_SUBFOLDER = f'globalRGBhisto_color_{bins}bins'
    
    # Ensure the output directory exists
    os.makedirs(os.path.join(OUT_FOLDER, OUT_SUBFOLDER), exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing GLOBAL histogram with {bins} bins per channel ({bins**3} total bins)")
    print(f"{'='*60}\n")
    
    # Accumulate histograms for averaging
    hist_accumulator = None
    img_count = 0
    
    for filename in os.listdir(os.path.join(DATASET_FOLDER, 'Images')):
        if filename.endswith(".bmp"):
            img_count += 1
            if img_count % 50 == 0:
                print(f"Processing file {img_count}: {filename}")
            
            img_path = os.path.join(DATASET_FOLDER, 'Images', filename)
            img = cv2.imread(img_path).astype(np.float64) / 255.0  # Normalize the image
            
            # Convert BGR to RGB
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            
            fout = os.path.join(OUT_FOLDER, OUT_SUBFOLDER, filename.replace('.bmp', '.mat'))
            
            # Global histogram (grid_size=1 means no spatial grid)
            hist = spatialGridHistogramDescriptor(img, grid_size=1, feature_type='color', bins_per_channel=bins, orientation_bins=8)
            # It is the same as the one below!!!!!
            # hist = globalColorHistogramDescriptor(img, bins_per_channel=bins)
            
            # Accumulate for average
            if hist_accumulator is None:
                hist_accumulator = hist.copy()
            else:
                hist_accumulator += hist
            
            # Save the descriptor to a .mat file
            sio.savemat(fout, {'Hist': hist})
    
    # Compute average histogram
    avg_hist = hist_accumulator / img_count
    all_descriptors[f'Global_{bins}bins'] = avg_hist
    print(f"Processed {img_count} images - Average histogram computed")

print("\nPart 1 Complete: All global histograms computed")


# ======================================== PART 2: SPATIAL GRID DESCRIPTORS ========================================
print("\n" + "="*70)
print("PART 2: COMPUTING SPATIAL GRID DESCRIPTORS")
print("="*70 + "\n")

SPATIAL_EXPERIMENTS = [
    # Different grid sizes with color
    {'name': 'spatialGrid_2x2_color_4bins', 'grid': 2, 'feature': 'color', 'color_bins': 4, 'orient_bins': 0},
    {'name': 'spatialGrid_3x3_color_4bins', 'grid': 3, 'feature': 'color', 'color_bins': 4, 'orient_bins': 0},
    {'name': 'spatialGrid_4x4_color_4bins', 'grid': 4, 'feature': 'color', 'color_bins': 4, 'orient_bins': 0},

    {'name': 'spatialGrid_2x2_color_8bins', 'grid': 2, 'feature': 'color', 'color_bins': 8, 'orient_bins': 0},
    {'name': 'spatialGrid_3x3_color_8bins', 'grid': 3, 'feature': 'color', 'color_bins': 8, 'orient_bins': 0},
    {'name': 'spatialGrid_4x4_color_8bins', 'grid': 4, 'feature': 'color', 'color_bins': 8, 'orient_bins': 0},

    {'name': 'spatialGrid_2x2_color_12bins', 'grid': 2, 'feature': 'color', 'color_bins': 12, 'orient_bins': 0},
    {'name': 'spatialGrid_3x3_color_12bins', 'grid': 3, 'feature': 'color', 'color_bins': 12, 'orient_bins': 0},
    {'name': 'spatialGrid_4x4_color_12bins', 'grid': 4, 'feature': 'color', 'color_bins': 12, 'orient_bins': 0},

    {'name': 'spatialGrid_2x2_color_16bins', 'grid': 2, 'feature': 'color', 'color_bins': 16, 'orient_bins': 0},
    {'name': 'spatialGrid_3x3_color_16bins', 'grid': 3, 'feature': 'color', 'color_bins': 16, 'orient_bins': 0},
    {'name': 'spatialGrid_4x4_color_16bins', 'grid': 4, 'feature': 'color', 'color_bins': 16, 'orient_bins': 0},
    

    # Texture with different angular quantization
    {'name': 'spatialGrid_2x2_texture_4orient', 'grid': 2, 'feature': 'texture', 'color_bins': 0, 'orient_bins': 4},
    {'name': 'spatialGrid_3x3_texture_4orient', 'grid': 3, 'feature': 'texture', 'color_bins': 0, 'orient_bins': 4},
    {'name': 'spatialGrid_4x4_texture_4orient', 'grid': 4, 'feature': 'texture', 'color_bins': 0, 'orient_bins': 4},

    {'name': 'spatialGrid_2x2_texture_8orient', 'grid': 2, 'feature': 'texture', 'color_bins': 0, 'orient_bins': 8},
    {'name': 'spatialGrid_3x3_texture_8orient', 'grid': 3, 'feature': 'texture', 'color_bins': 0, 'orient_bins': 8},
    {'name': 'spatialGrid_4x4_texture_8orient', 'grid': 4, 'feature': 'texture', 'color_bins': 0, 'orient_bins': 8},

    {'name': 'spatialGrid_2x2_texture_12orient', 'grid': 2, 'feature': 'texture', 'color_bins': 0, 'orient_bins': 12},
    {'name': 'spatialGrid_3x3_texture_12orient', 'grid': 3, 'feature': 'texture', 'color_bins': 0, 'orient_bins': 12},
    {'name': 'spatialGrid_4x4_texture_12orient', 'grid': 4, 'feature': 'texture', 'color_bins': 0, 'orient_bins': 12},

    {'name': 'spatialGrid_2x2_texture_16orient', 'grid': 2, 'feature': 'texture', 'color_bins': 0, 'orient_bins': 16},
    {'name': 'spatialGrid_3x3_texture_16orient', 'grid': 3, 'feature': 'texture', 'color_bins': 0, 'orient_bins': 16},
    {'name': 'spatialGrid_4x4_texture_16orient', 'grid': 4, 'feature': 'texture', 'color_bins': 0, 'orient_bins': 16},
    

    # Combined color + texture
    {'name': 'spatialGrid_2x2_both_4bins_4orient', 'grid': 2, 'feature': 'both', 'color_bins': 4, 'orient_bins': 4},
    {'name': 'spatialGrid_3x3_both_4bins_4orient', 'grid': 3, 'feature': 'both', 'color_bins': 4, 'orient_bins': 4},
    {'name': 'spatialGrid_4x4_both_4bins_4orient', 'grid': 4, 'feature': 'both', 'color_bins': 4, 'orient_bins': 4},

    {'name': 'spatialGrid_2x2_both_8bins_4orient', 'grid': 2, 'feature': 'both', 'color_bins': 8, 'orient_bins': 4},
    {'name': 'spatialGrid_3x3_both_8bins_4orient', 'grid': 3, 'feature': 'both', 'color_bins': 8, 'orient_bins': 4},
    {'name': 'spatialGrid_4x4_both_8bins_4orient', 'grid': 4, 'feature': 'both', 'color_bins': 8, 'orient_bins': 4},

    {'name': 'spatialGrid_2x2_both_4bins_8orient', 'grid': 2, 'feature': 'both', 'color_bins': 4, 'orient_bins': 8},
    {'name': 'spatialGrid_3x3_both_4bins_8orient', 'grid': 3, 'feature': 'both', 'color_bins': 4, 'orient_bins': 8},
    {'name': 'spatialGrid_4x4_both_4bins_8orient', 'grid': 4, 'feature': 'both', 'color_bins': 4, 'orient_bins': 8},

    {'name': 'spatialGrid_2x2_both_8bins_8orient', 'grid': 2, 'feature': 'both', 'color_bins': 8, 'orient_bins': 8},
    {'name': 'spatialGrid_3x3_both_8bins_8orient', 'grid': 3, 'feature': 'both', 'color_bins': 8, 'orient_bins': 8},
    {'name': 'spatialGrid_4x4_both_8bins_8orient', 'grid': 4, 'feature': 'both', 'color_bins': 8, 'orient_bins': 8},

    {'name': 'spatialGrid_2x2_both_12bins_12orient', 'grid': 2, 'feature': 'both', 'color_bins': 12, 'orient_bins': 12},
    {'name': 'spatialGrid_3x3_both_12bins_12orient', 'grid': 3, 'feature': 'both', 'color_bins': 12, 'orient_bins': 12},
    {'name': 'spatialGrid_4x4_both_12bins_12orient', 'grid': 4, 'feature': 'both', 'color_bins': 12, 'orient_bins': 12},

    {'name': 'spatialGrid_2x2_both_16bins_16orient', 'grid': 2, 'feature': 'both', 'color_bins': 16, 'orient_bins': 16},
    {'name': 'spatialGrid_3x3_both_16bins_16orient', 'grid': 3, 'feature': 'both', 'color_bins': 16, 'orient_bins': 16},
    {'name': 'spatialGrid_4x4_both_16bins_16orient', 'grid': 4, 'feature': 'both', 'color_bins': 16, 'orient_bins': 16},
]

for exp_num, exp in enumerate(SPATIAL_EXPERIMENTS, 1):
    OUT_SUBFOLDER = exp['name']
    
    # Ensure the output directory exists
    os.makedirs(os.path.join(OUT_FOLDER, OUT_SUBFOLDER), exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SPATIAL Experiment {exp_num}/{len(SPATIAL_EXPERIMENTS)}: {exp['name']}")
    print(f"{'='*60}")
    print(f"  Grid: {exp['grid']}x{exp['grid']}, Feature: {exp['feature']}")
    print(f"  Color bins: {exp['color_bins']}, Orientation bins: {exp['orient_bins']}")
    
    # Calculate descriptor size
    if exp['feature'] == 'color':
        desc_size = (exp['color_bins']**3) * (exp['grid']**2)
    elif exp['feature'] == 'texture':
        desc_size = exp['orient_bins'] * (exp['grid']**2)
    else:  # both
        desc_size = ((exp['color_bins']**3) + exp['orient_bins']) * (exp['grid']**2)
    print(f"  Expected descriptor size: {desc_size} dimensions\n")
    
    # Accumulate histograms for averaging
    hist_accumulator = None
    img_count = 0
    
    for filename in os.listdir(os.path.join(DATASET_FOLDER, 'Images')):
        if filename.endswith(".bmp"):
            img_count += 1
            if img_count % 50 == 0:
                print(f"  Processing image {img_count}: {filename}")
            
            img_path = os.path.join(DATASET_FOLDER, 'Images', filename)
            img = cv2.imread(img_path).astype(np.float64) / 255.0
            
            # Convert BGR to RGB
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            
            fout = os.path.join(OUT_FOLDER, OUT_SUBFOLDER, filename.replace('.bmp', '.mat'))
            
            # Spatial grid descriptor
            hist = spatialGridHistogramDescriptor(
                img, 
                grid_size=exp['grid'],
                feature_type=exp['feature'],
                bins_per_channel=exp['color_bins'],
                orientation_bins=exp['orient_bins']
            )
            
            # Accumulate for average
            if hist_accumulator is None:
                hist_accumulator = hist.copy()
            else:
                hist_accumulator += hist
            
            # Save the descriptor to a .mat file
            sio.savemat(fout, {'Hist': hist})
    
    # Compute average histogram
    avg_hist = hist_accumulator / img_count
    all_descriptors[exp['name']] = avg_hist
    print(f"  Processed {img_count} images - Average histogram computed")
    print(f"\nCompleted spatial experiment {exp_num}/{len(SPATIAL_EXPERIMENTS)}")

print("\nPart 2 Complete: All spatial descriptors computed")


# ============================================= COMBINED VISUALIZATION =============================================
print("\n" + "="*70)
print("CREATING COMBINED VISUALIZATIONS")
print("="*70 + "\n")

# Create output directory for plots
PLOT_FOLDER = os.path.join('descriptor_plots')
os.makedirs(PLOT_FOLDER, exist_ok=True)

# 1. Plot Global Histograms Comparison
plt.figure(figsize=(14, 6))
for bins in QUANTIZATION_LEVELS:
    key = f'Global_{bins}bins'
    hist = all_descriptors[key]
    plt.plot(range(len(hist)), hist, label=f'{bins} bins/channel ({bins**3} total)', linewidth=2)

plt.xlabel('Bin Index', fontsize=12)
plt.ylabel('Average Normalized Frequency', fontsize=12)
plt.title('Average Global RGB Color Histograms (Part 1)\nAveraged Across All Images', fontsize=14, fontweight='bold')
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(PLOT_FOLDER, 'part1_global_histograms_average.png'), dpi=300, bbox_inches='tight')
print("Saved: part1_global_histograms_average.png")
plt.close()

# 2. Create subplots for spatial descriptors by category
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Color-only descriptors
ax = axes[0]
color_exps = [exp for exp in SPATIAL_EXPERIMENTS if exp['feature'] == 'color']
for exp in color_exps[:12]:  # Select subset for clarity
    hist = all_descriptors[exp['name']]
    label = f"{exp['grid']}x{exp['grid']} grid, {exp['color_bins']} bins"
    ax.plot(range(len(hist)), hist, label=label, linewidth=1.5, alpha=0.8)
ax.set_xlabel('Feature Index', fontsize=11)
ax.set_ylabel('Average Value', fontsize=11)
ax.set_title('Spatial Color Descriptors (Part 2) - Averaged Across All Images', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3)

# Texture-only descriptors
ax = axes[1]
texture_exps = [exp for exp in SPATIAL_EXPERIMENTS if exp['feature'] == 'texture']
for exp in texture_exps[:12]:  # Select subset for clarity
    hist = all_descriptors[exp['name']]
    label = f"{exp['grid']}x{exp['grid']} grid, {exp['orient_bins']} orient"
    ax.plot(range(len(hist)), hist, label=label, linewidth=1.5, alpha=0.8)
ax.set_xlabel('Feature Index', fontsize=11)
ax.set_ylabel('Average Value', fontsize=11)
ax.set_title('Spatial Texture Descriptors (Part 2) - Averaged Across All Images', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3)

# Combined descriptors
ax = axes[2]
both_exps = [exp for exp in SPATIAL_EXPERIMENTS if exp['feature'] == 'both']
for exp in both_exps[:12]:  # Select subset for clarity
    hist = all_descriptors[exp['name']]
    label = f"{exp['grid']}x{exp['grid']}, {exp['color_bins']}c+{exp['orient_bins']}t"
    ax.plot(range(len(hist)), hist, label=label, linewidth=1.5, alpha=0.8)
ax.set_xlabel('Feature Index', fontsize=11)
ax.set_ylabel('Average Value', fontsize=11)
ax.set_title('Spatial Combined (Color+Texture) Descriptors (Part 2) - Averaged Across All Images', 
             fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_FOLDER, 'part2_spatial_descriptors_average.png'), dpi=300, bbox_inches='tight')
print("Saved: part2_spatial_descriptors_average.png")
plt.close()

# 3. Create mega comparison plot (selected descriptors from both parts)
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Top left: Global histograms
ax1 = fig.add_subplot(gs[0, 0])
for bins in [4, 8, 16, 32]:
    key = f'Global_{bins}bins'
    hist = all_descriptors[key]
    ax1.plot(range(len(hist)), hist, label=f'Global {bins} bins', linewidth=2)
ax1.set_title('Part 1: Global Histograms', fontsize=13, fontweight='bold')
ax1.set_xlabel('Bin Index')
ax1.set_ylabel('Average Frequency')
ax1.legend()
ax1.grid(alpha=0.3)

# Top right: Spatial color
ax2 = fig.add_subplot(gs[0, 1])
selected_color = [exp['name'] for exp in SPATIAL_EXPERIMENTS if exp['feature'] == 'color']
for name in selected_color:
    hist = all_descriptors[name]
    # Extract grid size and bins from name
    parts = name.split('_')
    grid = parts[1]
    bins = parts[3].replace('bins', '')
    ax2.plot(range(len(hist)), hist, label=f'{grid} {bins}bins', linewidth=1.5, alpha=0.7)
ax2.set_title('Part 2: Spatial Color (All)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Feature Index')
ax2.set_ylabel('Average Value')
ax2.legend(fontsize=8, ncol=2)
ax2.grid(alpha=0.3)

# Bottom left: Spatial texture
ax3 = fig.add_subplot(gs[1, 0])
selected_texture = [exp['name'] for exp in SPATIAL_EXPERIMENTS if exp['feature'] == 'texture']
for name in selected_texture:
    hist = all_descriptors[name]
    # Extract grid size and orientation bins from name
    parts = name.split('_')
    grid = parts[1]
    orient = parts[3].replace('orient', '')
    ax3.plot(range(len(hist)), hist, label=f'{grid} {orient}orient', linewidth=1.5, alpha=0.7)
ax3.set_title('Part 2: Spatial Texture (All)', fontsize=13, fontweight='bold')
ax3.set_xlabel('Feature Index')
ax3.set_ylabel('Average Value')
ax3.legend(fontsize=8, ncol=2)
ax3.grid(alpha=0.3)

# Bottom right: Combined
ax4 = fig.add_subplot(gs[1, 1])
selected_both = [exp['name'] for exp in SPATIAL_EXPERIMENTS if exp['feature'] == 'both']
for name in selected_both:
    hist = all_descriptors[name]
    # Extract grid size, color bins and orientation bins from name
    parts = name.split('_')
    grid = parts[1]
    bins = parts[3].replace('bins', '')
    orient = parts[4].replace('orient', '')
    ax4.plot(range(len(hist)), hist, label=f'{grid} {bins}c+{orient}t', linewidth=1.5, alpha=0.7)
ax4.set_title('Part 2: Spatial Combined (All)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Feature Index')
ax4.set_ylabel('Average Value')
ax4.legend(fontsize=8, ncol=2)
ax4.grid(alpha=0.3)

fig.suptitle('Complete Descriptor Analysis: Parts 1 & 2 Combined\nAverage Histograms Across All Images', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig(os.path.join(PLOT_FOLDER, 'combined_parts1and2_comparison.png'), dpi=300, bbox_inches='tight')
print("Saved: combined_parts1and2_comparison.png")
plt.close()

# ===== SUMMARY =====
print("\n" + "="*70)
print("ALL DESCRIPTORS COMPUTED AND VISUALIZED SUCCESSFULLY!")
print("="*70)
print(f"\nGlobal histograms: {len(QUANTIZATION_LEVELS)} configurations")
print(f"Spatial grid descriptors: {len(SPATIAL_EXPERIMENTS)} configurations")
print(f"Total: {len(QUANTIZATION_LEVELS) + len(SPATIAL_EXPERIMENTS)} descriptor sets")
print(f"\nAll histograms are AVERAGED across all images in the dataset")
print(f"\nGenerated visualizations:")
print("  1. part1_global_histograms_average.png - Global histogram comparison")
print("  2. part2_spatial_descriptors_average.png - Spatial descriptors by category")
print("  3. combined_parts1and2_comparison.png - Combined 4-panel comparison")
print("\nNext step: Run cvpr_visualsearch.py with different DESCRIPTOR_SUBFOLDER values")
print("="*70 + "\n")