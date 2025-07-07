import numpy as np
from scipy.ndimage import label, generate_binary_structure
from skimage.morphology import ball

def refine_lesion_mask(
    lesion_mask,
    wm_mask,
    gm_mask,
    csf_mask,
    brain_center_coord = (91, 109, 91)
    ):
    """
    Refines a binary lesion mask by iterating over detected lesion regions and applying user-defined selection criteria.
    Parameters
    ----------
    lesion_mask : np.ndarray
        A 3D binary array indicating the initial lesion mask (True/1 for lesion voxels).
    wm_mask : np.ndarray
        A 3D binary array indicating the white matter mask.
    gm_mask : np.ndarray
        A 3D binary array indicating the gray matter mask.
    csf_mask : np.ndarray
        A 3D binary array indicating the cerebrospinal fluid (CSF) mask.
    brain_center_coord : tuple of int
        Coordinates (x, y, z) representing the center of the brain, which can be used for spatial filtering.

    Returns
    -------
    refined_mask : np.ndarray
        A 3D binary array of the same shape as `lesion_mask`, where only the selected lesions are retained.
    Notes
    -----
    This function is designed to be completed by YOU. The section marked as "STUDENT SECTION" is where you should 
    implement your own logic to decide whether each detected lesion region should be kept or discarded. 
    You can use the provided parameters and masks to define your criteria, such as:
        - Minimum lesion size
        - Proximity to the brain center
        - Overlap with white matter, gray matter, or CSF
        - Shape or connectivity of the lesion
    You are encouraged to experiment with different conditions and thresholds to optimize lesion selection for your 
    specific dataset or research question.
    """

    refined_mask = np.zeros_like(lesion_mask, dtype=bool)
    structure = generate_binary_structure(3, 3)  # 26-neighbourhood
    labeled_mask, num_features = label(lesion_mask, structure=structure)

    for region_label in range(1, num_features + 1):
        region = (labeled_mask == region_label)

        # --- STUDENT SECTION START ---
        # Add your own conditions here to select or discard WM lesions.
        # For example:
        # if <your_condition>:
        #     continue
        # --- STUDENT SECTION END ---

        refined_mask[region] = 1

    return refined_mask

# Example usage:
# final_lesion_mask = refine_lesion_mask(initial_lesion_mask, wm_mask, gm_mask, csf_mask, (91, 109, 91))
