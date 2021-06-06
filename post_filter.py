import numpy as np
from scipy import ndimage

def post_filter(image, filter):
    """ Applies filter to the reconstructed image 
    
        filtered = post_filter(image, filter) returns the image filtered
        by filter
        
        denoise - remove noise by Gaussian blur followed by median averaging
        close - flatten image by grey dilation/erosion
        edge - apply the Sobel operator to detect edges
        unsharp - add overshoot to edges by unsharp masking
        
        Multistage filtering can be applied by setting filter to a list of 
        filters to be applied"""

    # check image type is broadcastable to array
    if not isinstance(image, np.ndarray):
        try:
            img = np.array(image)
        except:
            raise TypeError('image must be a 2D numpy array or broadcastable to a 2D array')
    else:
        img = np.copy(image)

    # check image is 2 dimensional
    if len(img.shape) != 2:
        raise TypeError('image must be a 2D array')

    # apply multistage filtering
    if isinstance(filter, (list, tuple)):

        # apply each filter in turn
        for subfilter in filter:
            img = post_filter(img, subfilter)
        return img

    # apply filter
    elif isinstance(filter, str):
        if filter is 'denoise':

            # remove noise by gaussian blur followed by median averaging
            blurred = ndimage.gaussian_filter(img, 3)
            return ndimage.median_filter(blurred, 5).astype('int')

        elif filter is 'close':
            
            # flatten image by grey dilation/erosion
            return ndimage.grey_closing(img, (5, 5)).astype('int')

        elif filter is 'edge':

            # apply the Sobel operator to detect edges
            sx = ndimage.sobel(img, axis=0, mode='constant')
            sy = ndimage.sobel(img, axis=1, mode='constant')
            edge = np.hypot(sx, sy)
            return np.interp(edge, (edge.min(), edge.max()), (-1024, 3071)).astype('int')

        elif filter is 'unsharp':

            # add overshoot to edges by unsharp masking
            blurred = ndimage.gaussian_filter(img, 3)
            filter_blurred = ndimage.gaussian_filter(blurred, 1)
            unsharp = blurred + 30 * (blurred - filter_blurred)
            return np.interp(unsharp, (unsharp.min(), unsharp.max()), (-1024, 3071)).astype('int')

        else:

            # raise error for invalid filter
            raise ValueError("filter '{}' not available. Possible filters are: 'denoise', 'close', 'edge' and 'unsharp'".format(filter))

    else:

        # raise error for invalid filter
        raise TypeError('filter must be a string or list of strings')