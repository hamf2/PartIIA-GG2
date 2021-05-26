from ct_detect import ct_detect
from attenuate import attenuate
import numpy as np
import scipy
from scipy import interpolate

def ct_calibrate(photons, material, sinogram, scale, correct=True):

	""" ct_calibrate convert CT detections to linearised attenuation
	sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
	in x (angles x samples) and returns a linear attenuation sinogram
	(angles x samples). photons is the source energy distribution, material is the
	material structure containing names, linear attenuation coefficients and
	energies in mev, and scale is the size of each pixel in x, in cm."""

	# Get dimensions and work out detection for just air of twice the side
	# length (has to be the same as in ct_scan.py)
	n = sinogram.shape[1]
	calibration = ct_detect(photons, material.coeff('Air'),  2 * scale * n)

	# perform calibration
	p = - np.log(sinogram / calibration)

	# apply beam hardening correction
	if correct:
		# create array of thicknesses and corresponding attenuations
		t = scale * np.arange(np.clip(n, 256, None))
		p_w = - np.log(ct_detect(photons, material.coeff('Water'), t) / calibration)

		# fit polynomial and evaluate for scan attenuations
		f = np.polynomial.polynomial.polyfit(p_w, t, 3)
		t_m = np.polynomial.polynomial.polyval(p, f)

		# apply scaling
		C = 0.243
		p = C * t_m

	return p