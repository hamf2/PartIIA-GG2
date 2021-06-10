import math
import numpy as np
from numpy.core.numeric import zeros_like
import numpy.matlib

def ramp_filter(sinogram, scale, alpha=None, window='cosine'):
	""" Ram-Lak filter with raised-cosine for CT reconstruction

	fs = ramp_filter(sinogram, scale) filters the input in sinogram (angles x samples)
	using a Ram-Lak filter.

	fs = ramp_filter(sinogram, scale, alpha, window) can be used to modify the Ram-Lak filter by a
	window function defined by alpha."""

	# set default alpha
	try:
		alpha = {'cosine': 1, 'sinc': 1, 'sumcos': 0.5}[window]
	except KeyError:
		if window in ['ram-lak', 'hamming', 'hann', 'shepp-logan']:
			alpha = {'ram-lak': 0, 'hamming': 0.54, 'hann': 0.5, 'shepp-logan': 1}[window]
			window = {'ram-lak': 'cosine', 'hamming': 'sumcos', 'hann': 'sumcos', 'shepp-logan': 'sinc'}[window]
		else:
			raise ValueError("window '{}' not available. Possible windows are: 'cosine', 'sinc' and 'sumcos'".format(window))

	# get input dimensions
	angles = sinogram.shape[0]
	n = sinogram.shape[1]

	#Set up filter to be at least twice as long as input
	m = np.ceil(np.log(2*n-1) / np.log(2))
	m = int(2 ** m)

	# define values for raised Ram-Lak filter (positive frequency only)
	if window is 'cosine':
		q_k = lambda k, a : (np.where(k == 0, 1, 0) * np.cos(math.pi / m) ** a / 6 + np.abs(k) * np.cos((math.pi * k) / m) ** a) / (m * scale)
	elif window is 'sinc':
		q_k = lambda k, a : (np.where(k == 0, 1, 0) * np.sinc(1 / m) ** a / 6 + np.abs(k) * np.sinc(k / m) ** a) / (m * scale)
	else:
		q_k = lambda k, a : (np.where(k == 0, 1, 0) * (a + (1 - a) * np.cos(2 * math.pi / m)) / 6 + np.abs(k) * (np.exp(-a) + (1 - np.exp(-a)) * np.cos(2 * math.pi * k/ m))) / (m * scale)

	q = q_k(np.arange(m//2 + 1), alpha)

	# apply filter to all angles
	print('Ramp filtering')
	ft = np.fft.rfft(sinogram, m, axis=1)
	filtered = np.fft.irfft(ft * q, axis= 1)[:, :n]
	
	return filtered