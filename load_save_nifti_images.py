import nibabel as nib

def load_nifti_image(file_path):

    """
    Loads a NIfTI image from the specified file path.

    Parameters:
        file_path (str): Path to the NIfTI (.nii or .nii.gz) file.

    Returns:
        data (numpy.ndarray): Image data array.
        affine (numpy.ndarray): Affine transformation matrix.
    
    Raises:
        ValueError: If the file is not a NIfTI image or if the file does not exist.
        TypeError: If the file path is not a string or is empty.

    Example:
        >>> T1_image, T1_affine = load_nifti_image('T1_image.nii.gz')
    """
    # Load the NIfTI image using nibabel
    if not file_path.endswith(('.nii', '.nii.gz')):
        raise ValueError("File must be a NIfTI image (.nii or .nii.gz)")
    if not isinstance(file_path, str):
        raise TypeError("File path must be a string")
    if not file_path:
        raise ValueError("File path cannot be empty")
    if not nib.filebasedimages.is_supported(file_path):
        raise ValueError("File format not supported or file does not exist")
    
    # Load the image data and affine transformation matrix
    img = nib.load(file_path)
    data = img.get_fdata()
    affine = img.affine
    return data, affine


def save_nifti_image(data, affine, file_path):
    """
    Saves a NIfTI image to the specified file path.

    Parameters:
        data (numpy.ndarray): Image data array.
        affine (numpy.ndarray): Affine transformation matrix, can be any of the previously loaded affine matrices.
        file_path (str): Path to save the NIfTI (.nii or .nii.gz) file.

    Raises:
        ValueError: If the file path is not a string or is empty.
        TypeError: If data is not a numpy array or affine is not a numpy array.

    Example:
        >>> save_nifti_image(data, affine, 'output_image.nii.gz')
    """
    if not isinstance(file_path, str):
        raise TypeError("File path must be a string")
    if not file_path:
        raise ValueError("File path cannot be empty")
    
    # Create a NIfTI image and save it
    img = nib.Nifti1Image(data, affine)
    nib.save(img, file_path)