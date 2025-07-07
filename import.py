# Generale imports which could be useful
import nibabel as nib
import numpy as np
from os.path import join
import matplotlib. pyplot as plt
from scipy.stats import norm
from scipy.ndimage import label, binary_dilation, generate_binary_structure, gaussian_filter1d
from scipy.interpolate import UnivariateSpline