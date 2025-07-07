import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt

def compute_histogram(data, bins=256):
    """
    Computes the histogram of the input data, excluding zero values.

    Parameters
    ----------
    data : array-like
        Input data from which to compute the histogram.
    bins : int, optional
        Number of bins to use for the histogram (default is 256).

    Returns
    -------
    counts : ndarray
        The number of samples in each bin.
    bin_centers : ndarray
        The center value of each bin.
    bin_width : float
        The width of each bin.

    Notes
    -----
    Zero values in the input data are excluded from the histogram calculation.
    """
    counts, bin_edges = np.histogram(data[data > 0], bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_centers[1] - bin_centers[0]
    return counts, bin_centers, bin_width

def smooth_histogram(counts, sigma=2):
    """
    Smooths a histogram using a Gaussian filter.

    Parameters:
        counts (array-like): The input histogram counts to be smoothed.
        sigma (float, optional): Standard deviation for Gaussian kernel. Default is 2.

    Returns:
        numpy.ndarray: The smoothed histogram counts.

    Notes:
        This function applies a one-dimensional Gaussian filter to the input histogram counts
        to reduce noise and smooth the distribution.
    """
    return gaussian_filter1d(counts, sigma=sigma)

def estimate_mean_std(bin_centers, smoothed_counts, data):
    """
    Estimate the mean (μ) and standard deviation (σ) of a distribution from histogram data using spline fitting.

    This function fits a univariate spline to the smoothed histogram counts (minus half the maximum value)
    to estimate the full width at half maximum (FWHM) of the distribution. The mean is estimated as the
    bin center corresponding to the peak of the smoothed counts. The standard deviation is estimated from
    the FWHM using the relation σ = FWHM / 2.355 (assuming a Gaussian shape). If the FWHM cannot be determined,
    the standard deviation is estimated as the standard deviation of the positive values in the input data.

    Args:
        bin_centers (np.ndarray): Array of bin center positions for the histogram.
        smoothed_counts (np.ndarray): Smoothed histogram counts corresponding to each bin center.
        data (np.ndarray): Original data array used to compute the histogram.

    Returns:
        tuple: A tuple (mu_spline, sigma_spline) where
            mu_spline (float): Estimated mean (peak position) of the distribution.
            sigma_spline (float): Estimated standard deviation of the distribution.
    """
    # Fit spline to the curve minus half the max (for FWHM)
    spline = UnivariateSpline(bin_centers, smoothed_counts - np.max(smoothed_counts)/2, s=0)
    roots = spline.roots()
    # μ estimated as the peak of the curve
    peak_index = np.argmax(smoothed_counts)
    mu_spline = bin_centers[peak_index]
    # σ estimated from FWHM
    if len(roots) >= 2:
        fwhm = roots[-1] - roots[0]
        sigma_spline = fwhm / 2.355
    else:
        sigma_spline = np.std(data[data > 0])  # fallback
    return mu_spline, sigma_spline

def compute_threshold(mu, sigma, gamma):
    """
    Computes a threshold value based on the mean, standard deviation, and a scaling factor.

    Args:
        mu (float): The mean value.
        sigma (float): The standard deviation.
        gamma (float): The scaling factor to adjust the threshold.

    Returns:
        float: The computed threshold value.
    """
    return mu + gamma * sigma

def create_lesion_mask(image, threshold):
    """
    Creates a binary mask identifying lesion areas in an image based on a threshold.

    Parameters:
        image (numpy.ndarray): The input image as a NumPy array.
        threshold (float or int): The intensity threshold value. Pixels with values greater than this threshold are considered part of the lesion.

    Returns:
        numpy.ndarray: A boolean array of the same shape as `image`, where True indicates pixels classified as lesion.
    """
    return image > threshold

def plot_histogram(bin_centers, counts, bin_width, smoothed_counts, threshold, mu):
    """
    Plots a histogram of intensity values along with a smoothed curve and threshold indicators.

    Parameters:
        bin_centers (array-like): The center values of each histogram bin.
        counts (array-like): The frequency counts for each bin.
        bin_width (float): The width of each histogram bin.
        smoothed_counts (array-like): Smoothed version of the histogram counts (e.g., from a spline or kernel density estimate).
        threshold (float): The intensity value used as a threshold for lesion detection, shown as a vertical line.
        mu (float): The mean intensity value (μ), shown as a vertical line for reference.

    Displays:
        A matplotlib figure with:
            - The histogram of intensity values (as bars).
            - The smoothed histogram curve.
            - Vertical lines indicating the threshold and mean intensity.
            - Appropriate labels, legend, grid, and tight layout for clarity.
    """
    plt.figure(figsize=(8, 4))
    plt.bar(bin_centers, counts, width=bin_width, color='royalblue', alpha=0.7, label='GM intensity histogram')
    plt.plot(bin_centers, smoothed_counts, label='Smoothed', color='blue')
    plt.axvline(x=threshold, color='limegreen', linestyle='--', label=f'Lesion Threshold = {threshold:.1f}')
    plt.axvline(mu, color='red', linestyle='--', label=f'μ ≈ {mu:.1f}')
    plt.title('Spline fit on FLAIR histogram in GM')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage:
# counts, bin_centers, bin_width = compute_histogram(flair_gm)
# smoothed_counts = smooth_histogram(counts)
# mu, sigma = estimate_mu_sigma(bin_centers, smoothed_counts, flair_gm)
# threshold = compute_threshold(mu, sigma, gamma)  # gamma is the parameter you need to find
# initial_lesion_mask = create_lesion_mask(flair, threshold)
# plot_histogram(bin_centers, counts, bin_width, smoothed_counts, threshold, mu)