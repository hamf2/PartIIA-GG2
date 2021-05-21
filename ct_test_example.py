
# these are the imports you are likely to need
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *

# create object instances
material = Material()
source = Source()

# define each end-to-end test here, including comments
# these are just some examples to get you started
# all the output should be saved in a 'results' directory

def test_1():
	# produces a phantom and reconstruction for geometric comparison

	# set test size and scale
	n, angles, scale = 256, 128, 0.01
	s = source.photon('100kVp, 3mm Al')

	# create a phantom and reconstruction
	p = ct_phantom(material.name, n, 3)
	y = scan_and_reconstruct(s, material, p, scale, angles)

	# save some meaningful results
	save_draw(y, 'results', 'test_1_image')
	save_draw(p, 'results', 'test_1_phantom')

	# the reconstruction should resemble the phantom

def test_2():
	# plots the 1D impulse response of the reconstruction process

	# set test size and scale
	n, angles, scale = 256, 128, 0.01
	s = source.photon('80kVp, 1mm Al')

	# create a phantom and reconstruction
	p = ct_phantom(material.name, n, 2)
	y = scan_and_reconstruct(s, material, p, scale, angles)

	# save some meaningful results
	save_plot(y[128,:], 'results', 'test_2_plot')

	# the impulse response should of the form of a sharp sinc function

def test_3():
	# calculates the mean reconstructed attenuation of tissue

	# set test size and scale
	n, angles, scale = 256, 128, 0.01
	s = fake_source(source.mev, 0.1, method='ideal')
	
	# create a phantom and reconstruction
	p = ct_phantom(material.name, n, 1)
	y = scan_and_reconstruct(s, material, p, scale, angles)

	# save some meaningful results
	with open('results/test_3_output.txt', mode='w') as f:
		f.write('Mean value is ' + str(np.mean(y[64:192, 64:192])))

	# the value should be approximately 0.203963689535233, the attenuation coefficient
	# of soft tissue at 0.07MeV

def test_4(hu=False):
	# compare the reconstructed and real attenuation coefficients
	
	# set test size and scale
	n, angles, scale = 256, 128, 0.01
	s = fake_source(source.mev, 0.1, method='ideal')

	o_str = ''
	for name in material.name[:9]:
		# create a phantom and reconstruction
		p = ct_phantom(material.name, n, 1, metal=name)
		y = scan_and_reconstruct(s, material, p, scale, angles)

		# get attenuation coefficients
		mu_E = material.coeff(name)[np.argmax(s)]
		mu_reconstruction = np.mean(y[n//4:3*n//4, n//4:3*n//4])

		# convert to HU if reconstruction in HU
		if hu:
			mu_E = 1000 * (mu_E - material.coeff('Water')[np.argmax(s)]) / material.coeff('Water')[np.argmax(s)]

		# record actual and expected coefficients
		o_str += '{0:<35} {1:<20} {2:>12}'.format('Mean value for ' + name + ' is ', mu_reconstruction, '(Expected: ' + str(mu_E) + ')\n')
	
	# save some meaningful results
	with open('results/test_4_output.txt', mode='w') as f:
		f.write('Test with image size: ' +str(n)+'x'+str(n)+ ' and scale: ' +str(scale)+ ' cm/pixel\n\n')
		f.write(o_str)

def test_5(hu=False):
	# creates an image of the absolute error in the reconstruction

	# set test size, angles, scale and source energy
	n, angles, scale, E = 256, 256, 0.01, 0.1
	s = fake_source(source.mev, E, method='ideal')

	# create a phantom and reconstruction
	p = ct_phantom(material.name, n, 7)
	y = scan_and_reconstruct(s, material, p, scale, angles)

	# convert the phantom indices to material coefficients at the source energy
	p_mu = np.zeros_like(p)
	for index, name in enumerate(material.name):
		if index in p:
			p_mu += np.where(p == index, material.coeff(name)[np.argmax(s)], 0)
	
	# convert to HU if reconstruction in HU
	if hu:
		p_mu = 1000 * (p_mu - material.coeff('Water')[np.argmax(s)]) / material.coeff('Water')[np.argmax(s)]

	# calculate square error for each pixel
	err = np.where(y >= (0, -1024)[hu], np.abs(p_mu - y), 0)

	# save some meaningful results
	save_draw(err, 'results', 'test_5_image')

	# an ideal reconstruction would have no error
	# the error is focussed around the edges of the large attenuations

def test_6(hu=False):
	# assesses the accuracy of a rounding-based material reconstruction

	# set test size, angles, scale and source energy
	n, angles, scale, E = 256, 256, 0.01, 0.1
	s = fake_source(source.mev, E, method='ideal')

	# create a phantom and reconstruction
	p = ct_phantom(material.name, n, 6)
	y = scan_and_reconstruct(s, material, p, scale, angles)

	# get material coefficients at the source energy
	mu_E = material.coeffs[:, np.argmax(s)].flatten()

	# convert to HU if reconstruction in HU
	if hu:
		mu_E = 1000 * (mu_E - material.coeff('Water')[np.argmax(s)]) / material.coeff('Water')[np.argmax(s)]

	# round each reconstructed value to the nearest material
	p_reconstruction = np.argmin(np.abs(np.subtract.outer(y, mu_E)), axis=-1)

	# find areas where the reconstruction identifies the correct materials
	correct = np.where(p_reconstruction == p, 1, 0)

	# save some meaningful results
	save_draw(correct, 'results', 'test_6_image')

	# an ideal reconstruction would yield a fully white image
	# errors could be reduced by using a reduced material set based on expected
	# materials (ie Breast Tissue wouldn't be expected in a hip scan)

def test_7(hu=False):
	# calculates the structural similarity index between the attenuation 
	# phantom and reconstruction

	# set test size, angles, scale, source energy and filter thickness
	n, angles, scale, E, t = 256, 256, 0.01, 0.1, 2
	s_i = fake_source(source.mev, E, material.coeff('Aluminium'), t, method='ideal')
	s_r = source.photon('{:d}kVp, {:d}mm Al'.format(int(1000 * E), t))

	# create a phantom and reconstructions
	p = ct_phantom(material.name, n, 5)
	y_i = scan_and_reconstruct(s_i, material, p, scale, angles)
	y_r = scan_and_reconstruct(s_r, material, p, scale, angles)

	# convert the phantom indices to material coefficients at the source energy
	p_mu = np.zeros_like(p)
	for index, name in enumerate(material.name):
		if index in p:
			p_mu += np.where(p == index, material.coeff(name)[np.argmax(s_i)], 0)
	
	# convert to HU if reconstruction in HU
	if hu:
		p_mu = 1000 * (p_mu - material.coeff('Water')[np.argmax(s_i)]) / material.coeff('Water')[np.argmax(s_i)]
		p_mu = np.clip(p_mu, -1024, 3071)

		# remap to 8-bit values
		y_i = np.interp(y_i, (-1024, 3071), (0, 255))
		y_r = np.interp(y_r, (-1024, 3071), (0, 255))
		p_mu = np.interp(p_mu, (-1024, 3071), (0, 255))
	else:
		y_i = np.clip(y_i, 0, None)
		y_r = np.clip(y_r, 0, None)

		# remap to 8-bit values
		low, high = np.min(np.concatenate((y_i, y_r, p_mu))), np.max(np.concatenate((y_i, y_r, p_mu)))
		y_i = np.interp(y_i, (low, high), (0, 255))
		y_r = np.interp(y_r, (low, high), (0, 255))
		p_mu = np.interp(p_mu, (low, high), (0, 255))

	# calculate the SSIM
	L = 2 ** 8 - 1
	c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
	cov = lambda x, y : np.mean((y - np.mean(y)) * (x - np.mean(x)))
	SSIM = lambda x, y : (2 * np.mean(y) * np.mean(x) + c1) * (2 * cov(x, y) + c2) / ((np.mean(y) ** 2 + np.mean(x) ** 2 + c1) * (np.var(y) + np.var(x) + c2))

	# save some meaningful results
	with open('results/test_7_output.txt', mode='w') as f:
		f.write('SSIM between the attenuation phantom and ideal source reconstruction is ' + str(SSIM(p_mu, y_i)) + '\n')
		f.write('SSIM between the attenuation phantom and real source reconstruction is  ' + str(SSIM(p_mu, y_r)) + '\n')
		f.write('SSIM between the ideal source reconstruction and real source reconstruction is ' + str(SSIM(y_i, y_r)))

	# identical images have an SSIM of 1, the possible range is -1 to 1
	# the reconstructions should therefore have values close to 1

# Run the various tests
print('Test 1')
test_1()
print('Test 2')
test_2()
print('Test 3')
test_3()
print('Test 4')
test_4()
print('Test 5')
test_5()
print('Test 6')
test_6()
print('Test 7')
test_7()