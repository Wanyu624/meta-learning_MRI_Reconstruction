import tensorflow as tf
import numpy as np
import sys
sys.path.append('../')
from scipy.io import savemat
import os
import scipy.misc
#from spectrum import fftshift
#from tensorflow.python.ops.signal.helper import fftshift
#from tensorflow.python.ops.signal.helper import ifftshift 
#from tensorflow.python import roll as _roll
#from tensorflow.python.framework import ops
#from tensorflow.python.util.tf_export import tf_export


def fft2c(img):
    """ Centered fft2 """
    return np.fft.fft2(img) / np.sqrt(img.shape[-2]*img.shape[-1])

def ifft2c(img):
    """ Centered ifft2 """
    return np.fft.ifft2(img) * np.sqrt(img.shape[-2]*img.shape[-1])

def mriAdjointOp(rawdata, coilsens, mask):
    """ Adjoint MRI Cartesian Operator """
    mask = np.expand_dims( mask.astype(np.float32), axis=1)
    return np.sum(ifft2c(rawdata * mask)*np.conj(coilsens), axis=1)

def mriForwardOp(img, coilsens, mask):
    """ Forward MRI Cartesian Operator """
    mask = np.expand_dims( mask.astype(np.float32), axis=1)
    img = np.expand_dims( img, axis=1)
  
    return fft2c(coilsens * img)*mask

def fftshift(x, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.
    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.
    Parameters
    ----------
    x : array_like, Tensor
        Input array.
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.
    Returns
    -------
    y : Tensor.
    """
    x = tf.convert_to_tensor(x)
    if axes is None:
        axes = tuple(range(tf.rank(x)))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[ax] // 2 for ax in axes]

    return tf.roll(x, shift, axes)
#
##@tf_export("signal.ifftshift")
def ifftshift(x, axes=None):
    """
    The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.
    Parameters
    ----------
    x : array_like, Tensor.
    axes : int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.
    Returns
    -------
    y : Tensor.
    """
    #x = ops.convert_to_tensor_v2(x)
    if axes is None:
        axes = tuple(range(tf.keras.backend.dim(x)))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]

    return tf.roll(x, shift, axes)


def ifftc2d(inp):
    """ Centered inverse 2d Fourier transform, performed on axis (-1,-2).
    """
    shape = tf.shape(inp)
    numel = shape[-2]*shape[-1]
    scale = tf.sqrt(tf.cast(numel, tf.float32))

    #out = fftshift(tf.ifft2d(ifftshift(inp, axes= None)), axes= None)
    out = tf.ifft2d(inp)
    out = tf.complex(tf.real(out)*scale, tf.imag(out)*scale)
    return out

def fftc2d(inp):
    """ Centered 2d Fourier transform, performed on axis (-1,-2).
    """
    shape = tf.shape(inp)
    numel = shape[-2]*shape[-1]
    scale = 1.0 / tf.sqrt(tf.cast(numel, tf.float32))

    #out = fftshift(tf.fft2d(ifftshift(inp, axes= None)), axes= None)
    out = tf.fft2d(inp)
    out = tf.complex(tf.real(out) * scale, tf.imag(out) * scale)
    return out

def removeFEOversampling(src):
    """ Remove Frequency Encoding (FE) oversampling.
        This is implemented such that they match with the DICOM images.
    """
    assert src.ndim >= 2
    nFE, nPE = src.shape[-2:]
    if nPE != nFE:
        return np.take(src, np.arange(int(nFE*0.25)+1, int(nFE*0.75)+1), axis=-2)
    else:
        return src
    
def removePEOversampling(src):
    """ Remove Phase Encoding (PE) oversampling. """
    nPE = src.shape[-1]
    nFE = src.shape[-2]
    PE_OS_crop = (nPE - nFE) / 2

    if PE_OS_crop == 0:
        return src
    else:
        return np.take(src, np.arange(int(PE_OS_crop)+1, nPE-int(PE_OS_crop)+1), axis=-1)

def removeFE(src):
    assert src.ndim >= 2
    nFE, nPE = src.shape[-2:]
    return np.take(src, np.arange(int(nFE*0.25)+1, int(nFE*0.75)+1), axis=-2)

def removePE(src):
    nPE = src.shape[-1]
    nFE = src.shape[-2]
    PE_OS_crop = (nPE - nFE) / 2

    return np.take(src, np.arange(int(PE_OS_crop)+1, nPE-int(PE_OS_crop)+1), axis=-1)


def ssim(input, target, ksize=11, sigma=1.5, L=1.0):
    def ssimKernel(ksize=ksize, sigma=sigma):
        if sigma == None:  # no gauss weighting
            kernel = np.ones((ksize, ksize, 1, 1)).astype(np.float32)
        else:
            x, y = np.mgrid[-ksize // 2 + 1:ksize // 2 + 1, -ksize // 2 + 1:ksize // 2 + 1]
            kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
            kernel = kernel[:, :, np.newaxis, np.newaxis].astype(np.float32)
        return kernel / np.sum(kernel)

    kernel = tf.Variable(ssimKernel(), name='ssim_kernel', trainable=False)
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC')
    mu2 = tf.nn.conv2d(target, kernel, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC')
    mu1_sqr = mu1 ** 2
    mu2_sqr = mu2 ** 2
    mu1mu2 = mu1 * mu2
    sigma1_sqr = tf.nn.conv2d(input * input, kernel, strides=[1, 1, 1, 1], padding='VALID',
                              data_format='NHWC') - mu1_sqr
    sigma2_sqr = tf.nn.conv2d(target * target, kernel, strides=[1, 1, 1, 1], padding='VALID',
                              data_format='NHWC') - mu2_sqr
    sigma12 = tf.nn.conv2d(input * target, kernel, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC') - mu1mu2
    ssim_maps = ((2.0 * mu1mu2 + C1) * (2.0 * sigma12 + C2)) / ((mu1_sqr + mu2_sqr + C1) *
                                                                (sigma1_sqr + sigma2_sqr + C2))
    return tf.reduce_mean(tf.reduce_mean(ssim_maps, axis=(1, 2, 3)))



def saveAsMat(img, filename, matlab_id, mat_dict=None):
    """ Save mat files with ndim in [2,3,4]

        Args:
            img: image to be saved
            file_path: base directory
            matlab_id: identifer of variable
            mat_dict: additional variables to be saved
    """
    assert img.ndim in [2, 3, 4]

    img_arg = img.copy()
    if img.ndim == 3:
        img_arg = np.transpose(img_arg, (1, 2, 0))
    elif img.ndim == 4:
        img_arg = np.transpose(img_arg, (2, 3, 0, 1))

    if mat_dict == None:
        mat_dict = {matlab_id: img_arg}
    else:
        mat_dict[matlab_id] = img_arg

    dirname = os.path.dirname(filename) or '.'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    savemat(filename, mat_dict)

    
def _normalize(img):
    """ Normalize image between [0, 1] """
    tmp = img - np.min(img)
    tmp /= np.max(tmp)
    return tmp

def contrastStretching(img, saturated_pixel=0.004):
    """ constrast stretching according to imageJ
    http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm"""
    values = np.sort(img, axis=None)
    nr_pixels = np.size(values)
    lim = int(np.round(saturated_pixel*nr_pixels))
    v_min = values[lim]
    v_max = values[-lim-1]
    img = (img - v_min)*(255.0)/(v_max - v_min)
    img = np.minimum(255.0, np.maximum(0.0, img))
    return img


def getContrastStretchingLimits(img, saturated_pixel=0.004):
    """ constrast stretching according to imageJ
    http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm"""
    values = np.sort(img, axis=None)
    nr_pixels = np.size(values)
    lim = int(np.round(saturated_pixel*nr_pixels))
    v_min = values[lim]
    v_max = values[-lim-1]
    return v_min, v_max

def normalize(img, v_min, v_max, max_int=255.0):
    """ normalize image to [0, max_int] according to image intensities [v_min, v_max] """
    img = (img - v_min)*(max_int)/(v_max - v_min)
    img = np.minimum(max_int, np.maximum(0.0, img))
    return img


def imsave(img, filepath, normalize=True):
    """ Save an image. """
    path = os.path.dirname(filepath) or '.'
    if not os.path.exists(path):
        os.makedirs(path)

    if img.dtype == np.complex64 or img.dtype == np.complex128:
        print('img is complex! Take absolute value.')
        img = np.abs(img)

    if normalize:
        img = _normalize(img)
        img *= 255.0
    #scipy.misc.imsave(filepath, img.astype(np.uint8))
    