import numpy as np
from attenuate import *
from ct_calibrate import *

def hu(p, material, reconstruction, scale):
	""" convert CT reconstruction output to Hounsfield Units
	calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into Hounsfield
	Units, using the material coefficients, photon energy p and scale given."""

	# use water to calibrate
	n = max(reconstruction.shape)
	water = material.coeff('Water')

	# put this through the same calibration process as the normal CT data
	calibration = ct_calibrate(p, material, np.array(ct_detect(p, water, scale * n)).reshape((1, 1)), scale)

	# use result to convert to hounsfield units
	# limit minimum to -1024, which is normal for CT data.
	hounsfield = 1000 * (reconstruction - calibration) / calibration
	hounsfield = np.clip(hounsfield, -1024, 3071)

	return hounsfield