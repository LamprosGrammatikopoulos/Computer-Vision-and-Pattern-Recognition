# ------------------------------------------
# | Full Name: Lampros Grammatikopoulos    |
# | University email: lg01302@surrey.ac.uk |
# | University number: 6918674             |
# ------------------------------------------

import numpy as np
import cv2

def globalColorHistogramDescriptor(img, bins_per_channel=8):
    """
    Computes a global color histogram descriptor using RGB quantization.
    
    Parameters:
    -----------
    img : numpy array
        Normalized RGB image with values in [0,1]
    bins_per_channel : int
        Number of bins per color channel (default: 8 means 8x8x8 = 512 bins total)
    
    Returns:
    --------
    F : numpy array
        Normalized histogram feature vector
    """

    # Quantize each channel
    # Map [0,1] to [0, bins_per_channel-1]
    quantized = np.floor(img * bins_per_channel).astype(int)
    
    # Clip to handle edge case where value = 1.0
    quantized = np.clip(quantized, 0, bins_per_channel - 1)
    
    # Flatten spatial dimensions
    r = quantized[:, :, 0].flatten()
    g = quantized[:, :, 1].flatten()
    b = quantized[:, :, 2].flatten()
    
    # Compute linear index for each pixel
    indices = r * (bins_per_channel ** 2) + g * bins_per_channel + b
    
    # Use numpy's bincount for efficiency
    hist = np.bincount(indices, minlength=bins_per_channel ** 3)
    
    # Normalize histogram
    hist = hist / np.sum(hist)
    
    return hist

def spatialGridHistogramDescriptor(img, grid_size=2, feature_type='color', bins_per_channel=8, orientation_bins=8):
    """
    Computes a spatial grid descriptor with color and/or texture features.
    
    Parameters:
    -----------
    img : numpy array
        Normalized RGB image with values in [0,1]
    grid_size : int
        Grid dimension (e.g., 2 means 2x2 = 4 cells, 3 means 3x3 = 9 cells)
    feature_type : str
        'color': RGB histogram only
        'texture': Edge orientation histogram only
        'both': Concatenate color + texture
    bins_per_channel : int
        Number of bins per color channel (for color features)
    orientation_bins : int
        Number of orientation bins (for texture features)
    
    Returns:
    --------
    F : numpy array
        Normalized concatenated feature vector from all grid cells
    """
    
    h, w = img.shape[:2]
    
    # Calculate cell dimensions
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    all_features = []
    
    # Iterate through grid cells
    for i in range(grid_size):
        for j in range(grid_size):
            # Extract cell
            y_start = i * cell_h
            y_end = (i + 1) * cell_h if i < grid_size - 1 else h
            x_start = j * cell_w
            x_end = (j + 1) * cell_w if j < grid_size - 1 else w
            
            cell = img[y_start:y_end, x_start:x_end]
            
            # Extract features based on type
            if feature_type == 'color':
                cell_features = compute_color_histogram(cell, bins_per_channel)
            elif feature_type == 'texture':
                cell_features = compute_texture_histogram(cell, orientation_bins)
            elif feature_type == 'both':
                color_feat = compute_color_histogram(cell, bins_per_channel)
                texture_feat = compute_texture_histogram(cell, orientation_bins)
                cell_features = np.concatenate([color_feat, texture_feat])
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")
            
            all_features.append(cell_features)
    
    # Concatenate all cell features
    descriptor = np.concatenate(all_features)
    
    # Normalize the entire descriptor
    descriptor = descriptor / (np.sum(descriptor) + 1e-10)
    
    return descriptor


def compute_color_histogram(cell, bins_per_channel):
    """
    Compute RGB color histogram for a single cell.
    
    Parameters:
    -----------
    cell : numpy array
        Image cell with values in [0,1]
    bins_per_channel : int
        Number of bins per color channel
    
    Returns:
    --------
    hist : numpy array
        Color histogram
    """
    # Quantize each channel
    quantized = np.floor(cell * bins_per_channel).astype(int)
    quantized = np.clip(quantized, 0, bins_per_channel - 1)
    
    # Flatten spatial dimensions
    r = quantized[:, :, 0].flatten()
    g = quantized[:, :, 1].flatten()
    b = quantized[:, :, 2].flatten()
    
    # Compute linear index for each pixel
    indices = r * (bins_per_channel ** 2) + g * bins_per_channel + b
    
    # Create histogram
    hist = np.bincount(indices, minlength=bins_per_channel ** 3)
    
    # Normalize
    hist = hist.astype(float) / (np.sum(hist) + 1e-10)
    
    return hist


def compute_texture_histogram(cell, orientation_bins):
    """
    Compute edge orientation histogram for a single cell using gradients.
    
    Parameters:
    -----------
    cell : numpy array
        Image cell with values in [0,1]
    orientation_bins : int
        Number of orientation bins (e.g., 8 = 45° per bin)
    
    Returns:
    --------
    hist : numpy array
        Edge orientation histogram
    """
    # Convert to grayscale if color
    if len(cell.shape) == 3:
        gray = cv2.cvtColor((cell * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(float) / 255.0
    else:
        gray = cell
    
    # Compute gradients using Sobel
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude and orientation
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x)  # Range: [-pi, pi]
    
    # Convert orientation to [0, 2*pi]
    orientation = orientation % (2 * np.pi)
    
    # Quantize orientation into bins
    bin_size = 2 * np.pi / orientation_bins
    orientation_indices = np.floor(orientation / bin_size).astype(int)
    orientation_indices = np.clip(orientation_indices, 0, orientation_bins - 1)
    
    # Flatten
    orientation_indices = orientation_indices.flatten()
    magnitude = magnitude.flatten()
    
    # Weight histogram by gradient magnitude
    hist = np.zeros(orientation_bins)
    for idx, mag in zip(orientation_indices, magnitude):
        hist[idx] += mag
    
    # Normalize
    hist = hist / (np.sum(hist) + 1e-10)
    
    return hist
