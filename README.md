# README

## MS White Matter Lesion Segmentation

This repository provides Python scripts for segmenting Multiple Sclerosis (MS) white matter (WM) lesions using a thresholding method based on the intensity histogram of FLAIR MRI images within gray matter (GM) regions.

Please, open the Colab Notebook at the following link and make your own copy

https://colab.research.google.com/drive/1WuS-tad9hAAYFOGYuOx_PWlNKPyesefH?usp=sharing

## Instructions for Summer School Students

### 1. Prerequisites

- **Clone the Repository:**  
    In your Google Colab notebook, run:  

    ```python
    !git clone <repository_url>```

    ```%cd <repository_folder>
    ```

    Replace `<repository_url>` with the actual URL of this repository and `<repository_folder>` with the cloned folder name.

### 2. Input Data

For each of the 5 subjects, you will need the following files (all in NIfTI format, `.nii` or `.nii.gz`):

- **T1-weighted MRI Image**
- **T2-weighted MRI Image**
- **FLAIR MRI Image**
- **Ground Truth Segmentation Mask** (manual lesion annotation)
- **Tissue Segmentation Masks** (e.g., gray matter, white matter, CSF masks)

### 3. File Structure

- **Main script:**  
    You will need to create a main script (e.g., `segment_lesions.py`) that performs the segmentation by utilizing the provided supporting functions. The example commands and function descriptions below will help you compose this script.

- **Supporting functions:**  

    - `load_nifti_image(file_path)` — Loads a NIfTI image from `load_and_save_nifti_images.py`.  
        *Purpose:* Reads image data and header (affine matrix) from disk.

    - `save_nifti_image(data, affine, file_path)` — Saves a NIfTI image to the specified path from `load_and_save_nifti_images.py`.  
        *Purpose:* Writes image data and affine transformation to disk.

    - `compute_histogram(data, bins=256)` — Computes the histogram of input data, excluding zero values, from `histogram_and_thresholding.py`.  
        *Purpose:* Generates histogram counts and bin centers for intensity distribution analysis.

    - `smooth_histogram(counts, sigma=2)` — Smooths a histogram using a Gaussian filter from `histogram_and_thresholding.py`.  
        *Purpose:* Reduces noise in histogram counts for more robust analysis.

    - `estimate_mean_std(bin_centers, smoothed_counts, data)` — Estimates the mean and standard deviation from smoothed histogram data using spline fitting from `histogram_and_thresholding.py`.  
        *Purpose:* Determines key statistical parameters of the intensity distribution.

    - `compute_threshold(mu, sigma, gamma)` — Computes a threshold value based on mean, standard deviation, and a scaling factor from `histogram_and_thresholding.py`.  
        *Purpose:* Calculates the intensity threshold for lesion segmentation.

    - `create_lesion_mask(image, threshold)` — Creates a binary mask identifying lesion areas based on a threshold from `histogram_and_thresholding.py`.  
        *Purpose:* Applies the calculated threshold to generate an initial lesion mask.

    - `refine_lesion_mask(lesion_mask, wm_mask, gm_mask, csf_mask, brain_center_coord)` — Refines a binary lesion mask by iterating over detected lesion regions and applying user-defined selection criteria from `lesion_refinement.py`.  
        *Purpose:* Allows for post-processing and filtering of the initial lesion mask (students are to implement their own criteria in the "STUDENT SECTION").

    - `evaluate_segmentation(pred_mask, gt_mask)` — Evaluates segmentation performance by calculating True Positive Fraction, False Positive Fraction, and Dice coefficient from `segmentation_evaluation.py`.  
        *Purpose:* Quantifies the accuracy and overlap of the predicted lesion mask against the ground truth.

    - `plot_mri_images(T1, T2, flair, gt, slice_idx)` — Plots T1, T2, FLAIR MRI images, and ground truth segmentation mask from `plotting_functions.py`.  
        *Purpose:* Visualizes different MRI modalities and the ground truth for a specified slice.

    - `plot_brain_tissue_masks(wm_mask, gm_mask, csf_mask, slice_idx)` — Plots white matter, gray matter, and CSF segmentation masks from `plotting_functions.py`.  
        *Purpose:* Visualizes individual tissue masks for a specified slice.

    - `plot_flair_and_segmentations(flair, initial_lesion_mask, gt_mask, slice_idx)` — Plots FLAIR, obtained segmentation, and ground truth masks from `plotting_functions.py`.  
        *Purpose:* Compares the original FLAIR image with the initial segmentation and ground truth for a specified slice.

    - `plot_histogram(bin_centers, counts, bin_width, smoothed_counts, threshold, mu)` — Plots a histogram with smoothed curve and threshold indicators from `histogram_and_thresholding.py`.  
        *Purpose:* Provides a visual representation of the histogram analysis and threshold selection.

### 5. Output

- **Output File:**  
    A binary NIfTI mask (`.nii.gz`) highlighting the segmented WM lesions.

### 6. Customization

- **Thresholding Parameters:**  
    The core thresholding parameters are controlled by the `compute_threshold` function and the `gamma` parameter. You will adjust this as part of composing your main script.

### 7. Support

- For questions, contact the course instructors or refer to the comments in the respective Python files for detailed explanations of each function.

---

**Note:** Always visually inspect the segmentation results to ensure accuracy.  
For detailed function locations, refer to the comments in the respective files or use your code editor's search feature.
