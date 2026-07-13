------------------------------------------
| Full Name: Lampros Grammatikopoulos    |
------------------------------------------

General Instrustions & setup:
1) Use a windows 11 computer with as much ram as possible (preferably at least 16GB) and a fast CPU.
2) Install the latest version of python from: https://www.python.org/.
3) Unzip the contents of this zip in a folder.
4) Place the file called 'msrc_objcategimagedatabase_v2' with all its contents, in the same folder with the code files.
5) Open a terminal and install the packages required using the command: pip install requirements.txt


Requirement 1 and 2:
Requirement 1 and 2 are implemented in files cvpr_computedescriptors.py, descriptorFunctions.py, cvpr_visualsearch.py, and cvpr_compare.py.
1) In main function of cvpr_computedescriptors.py modify the list "QUANTIZATION_LEVELS" to set your desired number of color bins for G.C.H. descriptors.
2) in main function of cvpr_computedescriptors.py modify the list of dictionaries "SPATIAL_EXPERIMENTS" to set your desired number of color bins, 
   levels of angular quantization, and spatial grid size for the combined G.C.H., E.O.H., and Spatial Grid descriptors. 
   Be precise on the dictionaries format! They have to follow the following format:
   {'name': 'spatialGrid_2x2_color_4bins', 'grid': 2, 'feature': 'color', 'color_bins': 4, 'orient_bins': 0}
   Follow the examples in the code to add more descriptors or comment out if you want to calculate less of them.
3) Run cvpr_computedescriptors.py
4) An output folder called "descriptors" will be exported and will contain all the descriptor files.
5) An output folder called "descriptor_plots" will be exported and will contain all the descriptor plot files.
6) In main function of cvpr_visualsearch.py modify the list "DISTANCE_METRICS" to set the distance metric to "euclidean" only.
7) In main function of cvpr_visualsearch.py modify the list "DESCRIPTOR_SUBFOLDERS" to set the experiments to G.C.H. experiments only, 
   like 'globalRGBhisto_color_4bins'.
8) If you want to run the experiments on specific query image then in cvpr_visualsearch.py modify the line: 
   query_idx = np.random.randint(0, len(ALLFEAT)) -----> query_idx = {int}.
9) Run cvpr_visualsearch.py
10) An output folder called "visual_search_exports" will be exported and will contain all the plot files from all experiments.
11) Rename folder "visual_search_exports" before running next experiment so it doesn't get replaced.


Requirement 3:
Requirement 3 is implemented in files cvpr_visualsearch.py and cvpr_compare.py.
1) In main function of cvpr_visualsearch.py modify the list "DISTANCE_METRICS" to set the distance metric to "euclidean" only.
2) In main function of cvpr_visualsearch.py modify the list "DESCRIPTOR_SUBFOLDERS" to set the descriptors that will be used,
   like the following formats:
   'globalRGBhisto_color_4bins'         -> for color bins only,
   'spatialGrid_2x2_color_4bins'        -> for color bins with spatial grid,
   'spatialGrid_3x3_texture_8orient'    -> for angular quantization levels and spactial grid,
   'spatialGrid_4x4_both_8bins_8orient' -> for color bins, angular quantization levels, and spactial grid.
3) If you want to run the experiments on specific query image then in cvpr_visualsearch.py modify the line: 
   query_idx = np.random.randint(0, len(ALLFEAT)) -----> query_idx = {int}.
4) Run cvpr_visualsearch.py
5) An output folder called "visual_search_exports" will be exported and will contain all the plot files from all experiments.
6) Rename folder "visual_search_exports" before running next experiment so it doesn't get replaced.


Requirement 4:
Requirement 4 is implemented in files run_pca_comparision.py and cvpr_pca_utils.py.
1) In main function of run_pca_comparision.py modify the list "DESCRIPTOR_SUBFOLDERS" to set the descriptors that will be used,
   like the following formats:
   'globalRGBhisto_color_4bins'         -> for color bins only,
   'spatialGrid_2x2_color_4bins'        -> for color bins with spatial grid,
   'spatialGrid_3x3_texture_8orient'    -> for angular quantization levels and spactial grid,
   'spatialGrid_4x4_both_8bins_8orient' -> for color bins, angular quantization levels, and spactial grid.
2) In main function of run_pca_comparision.py modify the list "EXPERIMENTS" to set the PCA experiments that willl be used,
   like the following format:
   {'name': 'Baseline (Euclidean)', 'pca': False, 'distance': 'euclidean'},
   {'name': 'PCA 70% (Euclidean)', 'pca': True, 'variance': 0.70, 'distance': 'euclidean'},
   {'name': 'Baseline (Mahalanobis)', 'pca': False, 'distance': 'mahalanobis'},
   {'name': 'PCA 70% (Mahalanobis)', 'pca': True, 'variance': 0.70, 'distance': 'mahalanobis'},
3) Run run_pca_comparision.py
4) An output folder called "pca_exports" will be exported and will contain all the plot files from all experiments.
5) Also, an output folder called "pca_descriptors" will be exported and will contain all the PCA descriptor files.
6) Rename folder "pca_descriptors" before running next experiment so it doesn't get replaced.


Requirement 5:
Requirement 5 is implemented in files cvpr_visualsearch.py and cvpr_compare.py.
1) In main function of cvpr_visualsearch.py modify the list "DISTANCE_METRICS" to set the distance metric to 2 or more distance metrics available.
2) In main function of cvpr_visualsearch.py modify the list "DESCRIPTOR_SUBFOLDERS" to set the descriptors that will be used,
   like the following formats:
   'globalRGBhisto_color_4bins'         -> for color bins only,
   'spatialGrid_2x2_color_4bins'        -> for color bins with spatial grid,
   'spatialGrid_3x3_texture_8orient'    -> for angular quantization levels and spactial grid,
   'spatialGrid_4x4_both_8bins_8orient' -> for color bins, angular quantization levels, and spactial grid.
3) If you want to run the experiments on specific query image then in cvpr_visualsearch.py modify the line: 
   query_idx = np.random.randint(0, len(ALLFEAT)) -----> query_idx = {int}.
4) Run cvpr_visualsearch.py
5) An output folder called "visual_search_exports" will be exported and will contain all the plot files from all experiments.
6) Rename folder "visual_search_exports" before running next experiment so it doesn't get replaced.
