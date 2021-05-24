from ct_scan import *
from ct_calibrate import *
from ct_lib import *
from ramp_filter import *
from back_project import *
from hu import *

def scan_and_reconstruct(photons, material, phantom, scale, angles, mas=10000, alpha=0.001):

	""" Simulation of the CT scanning process
		reconstruction = scan_and_reconstruct(photons, material, phantom, scale, angles, mas, alpha)
		takes the phantom data in phantom (samples x samples), scans it using the
		source photons and material information given, as well as the scale (in cm),
		number of angles, time-current product in mas, and raised-cosine power
		alpha for filtering. The output reconstruction is the same size as phantom."""


	# convert source (photons per (mas, cm^2)) to photons

	# create sinogram from phantom data, with received detector values
	sinogram = ct_scan(photons, material, phantom, scale, angles, mas)

	# convert detector values into calibrated attenuation values
	sinogram = ct_calibrate(photons, material, sinogram, scale)

	# Ram-Lak
	sinogram = ramp_filter(sinogram, scale, alpha)

	# Back-projection
	reconstruction = back_project(sinogram)

	# convert to Hounsfield Units
	reconstruction = hu(photons, material, reconstruction, scale)

	return reconstruction