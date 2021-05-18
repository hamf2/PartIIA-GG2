import math
import numpy as np
from numpy.core.numeric import zeros_like
import numpy.matlib

def ramp_filter(sinogram, scale, alpha=0.001):
	""" Ram-Lak filter with raised-cosine for CT reconstruction

	fs = ramp_filter(sinogram, scale) filters the input in sinogram (angles x samples)
	using a Ram-Lak filter.

	fs = ramp_filter(sinogram, scale, alpha) can be used to modify the Ram-Lak filter by a
	cosine raised to the power given by alpha."""

	# get input dimensions
	angles = sinogram.shape[0]
	n = sinogram.shape[1]

	#Set up filter to be at least twice as long as input
	m = np.ceil(np.log(2*n-1) / np.log(2))
	m = int(2 ** m)
	q_k = lambda k, a : (np.where(k == 0, 1, 0) * np.cos(math.pi / n) ** a / 6 + np.where(k <= n//2, k, 0) * np.cos((math.pi * np.where(k <= n//2, k, 0)) / n) ** a) / (n * scale)
	q = q_k(np.arange(m//2 + 1), alpha)

	# apply filter to all angles
	print('Ramp filtering')
	ft = np.fft.rfft(sinogram, m, axis=1)
	filtered = np.fft.irfft(ft * q, axis= 1)[:, :n]
	
	return filtered