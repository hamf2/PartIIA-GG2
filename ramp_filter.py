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
	ft_q = lambda k, a : (np.where(k == 0, 1, 0) * np.cos(math.pi / m) ** a + 6 * np.abs(k) * np.cos((k * math.pi) / m) ** a) / (6 * scale * m)
	q = np.real(np.fft.ifft(ft_q(np.arange(m) - m/2, alpha)))

	# apply filter to all angles
	print('Ramp filtering')
	filtered = np.zeros((angles, n))
	for angle in range(angles):
		filtered[angle] = np.convolve(sinogram[angle], q)[:n]
	
	return sinogram