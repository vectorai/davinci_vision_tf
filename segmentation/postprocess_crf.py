import pydensecrf.densecrf as dcrf
import numpy as np

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary

import skimage.io as io

def process(final_probabilities,image,iters):

	softmax = final_probabilities.squeeze()
	print softmax.shape
	processed_probabilities = softmax.transpose((2, 0, 1))

	# The input should be the negative of the logarithm of probability values
	# Look up the definition of the softmax_to_unary for more information
	unary = softmax_to_unary(processed_probabilities)

	# The inputs should be C-continious -- we are using Cython wrapper
	unary = np.ascontiguousarray(unary)

	d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 2)

	d.setUnaryEnergy(unary)

	# This potential penalizes small pieces of segmentation that are
	# spatially isolated -- enforces more spatially consistent segmentations
	feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

	d.addPairwiseEnergy(feats, compat=3,
	                    kernel=dcrf.DIAG_KERNEL,
	                    normalization=dcrf.NORMALIZE_SYMMETRIC)

	# This creates the color-dependent features --
	# because the segmentation that we get from CNN are too coarse
	# and we can use local color features to refine them
	feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
	                                   img=image, chdim=2)

	d.addPairwiseEnergy(feats, compat=10,
	                     kernel=dcrf.DIAG_KERNEL,
	                     normalization=dcrf.NORMALIZE_SYMMETRIC)
	return d.inference(iters)
