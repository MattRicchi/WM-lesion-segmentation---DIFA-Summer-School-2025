import matplotlib.pyplot as plt

def plot_mri_images(T1, T2, flair, gt, slice_idx=91):
    """
    Plots T1, T2, FLAIR MRI images and ground truth segmentation mask for a given slice.

    Parameters:
        T1 (ndarray): T1-weighted MRI image.
        T2 (ndarray): T2-weighted MRI image.
        flair (ndarray): FLAIR MRI image.
        gt (ndarray): Ground truth segmentation mask.
        slice_idx (int): Index of the slice to display.
    """
    fig = plt.figure(figsize=(15, 12))
    plt.axis('off')

    rows, columns = 2, 2

    # Add T1 image
    fig.add_subplot(rows, columns, 1)
    plt.imshow(T1[:, :, slice_idx], cmap='gray')
    plt.title("$T_1$ weighted Image", fontsize=16)

    # Add T2 image
    fig.add_subplot(rows, columns, 2)
    plt.imshow(T2[:, :, slice_idx], cmap='gray')
    plt.title("$T_2$ weighted Image", fontsize=16)

    # Add FLAIR image
    fig.add_subplot(rows, columns, 3)
    plt.imshow(flair[:, :, slice_idx], cmap='gray')
    plt.title("FLAIR Image", fontsize=16)

    # Add segmentation mask
    fig.add_subplot(rows, columns, 4)
    plt.imshow(gt[:, :, slice_idx], cmap='gray')
    plt.title("Segmentation Mask", fontsize=16)

    plt.show()


def plot_brain_tissue_masks(wm_mask, gm_mask, csf_mask, slice_idx=91):
    """
    Plots white matter, gray matter, and CSF segmentation masks for a given slice.

    Parameters:
        wm_mask (ndarray): White matter segmentation mask.
        gm_mask (ndarray): Gray matter segmentation mask.
        csf_mask (ndarray): CSF segmentation mask.
        slice_idx (int): Index of the slice to display.
    """
    fig = plt.figure(figsize=(15, 6))
    plt.axis('off')

    rows, columns = 1, 3

    # Add White Matter mask
    fig.add_subplot(rows, columns, 1)
    plt.imshow(wm_mask[:, :, slice_idx], cmap='gray')
    plt.title("White Matter Segmentation", fontsize=16)

    # Add Gray Matter mask
    fig.add_subplot(rows, columns, 2)
    plt.imshow(gm_mask[:, :, slice_idx], cmap='gray')
    plt.title("Gray Matter Segmentation", fontsize=16)

    # Add CSF mask
    fig.add_subplot(rows, columns, 3)
    plt.imshow(csf_mask[:, :, slice_idx], cmap='gray')
    plt.title("CSF Segmentation", fontsize=16)

    plt.show()


def plot_flair_and_segmentations(flair, initial_lesion_mask, gt_mask, slice_idx=91):
    """
    Plots the original FLAIR image, the obtained segmentation, and the ground truth segmentation mask for a given slice.

    Parameters:
        flair (ndarray): FLAIR MRI image.
        initial_lesion_mask (ndarray): Obtained segmentation mask.
        gt_mask (ndarray): Ground truth segmentation mask.
        slice_idx (int): Index of the slice to display.
    """
    fig = plt.figure(figsize=(15, 6))
    plt.axis('off')

    rows, columns = 1, 3

    # Add FLAIR image at the 1st position
    fig.add_subplot(rows, columns, 1)
    plt.imshow(flair[:, :, slice_idx], cmap='gray')
    plt.title("FLAIR Image", fontsize=16)

    # Add obtained segmentation at the 2nd position
    fig.add_subplot(rows, columns, 2)
    plt.imshow(initial_lesion_mask[:, :, slice_idx], cmap='gray')
    plt.title("Initial Segmentation", fontsize=16)

    # Add ground truth segmentation at the 3rd position
    fig.add_subplot(rows, columns, 3)
    plt.imshow(gt_mask[:, :, slice_idx], cmap='gray')
    plt.title("Ground Truth Segmentation", fontsize=16)

    plt.show()