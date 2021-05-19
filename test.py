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

def test_1():
    # initial conditions: circle of material scanned by fake ideal source
    # set test size and scale
    n, scale = 128, 0.01
    s = fake_source(source.mev, 0.1, method='ideal')

    with open('test_output/test_1_output.txt', mode='w') as f:
        f.write('Test with image size: ' +str(n)+'x'+str(n)+ ' and scale: ' +str(scale)+ ' cm/pixel\n\n')

        for nmetal in material.name[:9]:
            p = ct_phantom(material.name, n, 1, metal=nmetal)
            y = scan_and_reconstruct(s, material, p, scale, n)

	        # save some meaningful results

            o_str = '{0:<35} {1:<20} {2:>12}'.format('Mean value for ' + nmetal + ' is ', str(np.mean(y[n//4:3*n//4, n//4:3*n//4])),
             '(Expected: ' + str(material.coeffs[material.name.index(nmetal)][np.argmax(s)] - material.coeffs[0][np.argmax(s)]) + ')\n')
            f.write(o_str)


# create output directory and run tests
if(not(os.path.exists('test_output'))):os.mkdir('test_output')
print('Test 1')
test_1()