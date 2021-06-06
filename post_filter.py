import numpy as np
from scipy import ndimage

def post_filter(image, filter):
    """ Applies filter to the reconstructed image 
    
        filtered = post_filter(image, filter) returns the image filtered
        by filter
        
        denoise - remove noise by Gaussian blur followed by median averaging
        close - flatten image by grey dilation/erosion
        edge - apply the Sobel operator to detect edges
        unsharp - add overshoot to edges by unsharp masking"""

    if filter is 'denoise':

        # remove noise by gaussian blur followed by median averaging
        blurred = ndimage.gaussian_filter(image, 3)
        return ndimage.median_filter(image, 5).astype('int')

    elif filter is 'close':
        
        # flatten image by grey dilation/erosion
        return ndimage.grey_closing(image, (5, 5)).astype('int')

    elif filter is 'edge':

        # apply the Sobel operator to detect edges
        sx = ndimage.sobel(image, axis=0, mode='constant')
        sy = ndimage.sobel(image, axis=1, mode='constant')
        edge = np.hypot(sx, sy)
        return np.interp(edge, (edge.min(), edge.max()), (-1024, 3071)).astype('int')

    elif filter is 'unsharp':

        # add overshoot to edges by unsharp masking
        blurred = ndimage.gaussian_filter(image, 3)
        filter_blurred = ndimage.gaussian_filter(blurred, 1)
        unsharp = blurred + 30 * (blurred - filter_blurred)
        return np.interp(unsharp, (unsharp.min(), unsharp.max()), (-1024, 3071)).astype('int')

    else:

        # return image for invalid filter
        print("Invalid filter, filters available are: 'denoise', 'close', 'edge' and 'unsharp'")
        return image