import numpy as np
from scipy import ndimage

def post_filter(image, filter):

    if filter is 'edge':
        sx = ndimage.sobel(image, axis=0, mode='constant')
        sy = ndimage.sobel(image, axis=1, mode='constant')
        edge = np.hypot(sx, sy)
        return np.interp(edge, (edge.min(), edge.max()), (-1024, 3071))
    elif filter is 'unsharp':
        blurred = ndimage.gaussian_filter(image, 3)
        filter_blurred = ndimage.gaussian_filter(blurred, 1)
        unsharp = blurred + 30 * (blurred - filter_blurred)
        return np.interp(unsharp, (unsharp.min(), unsharp.max()), (-1024, 3071))
    else:
        return image