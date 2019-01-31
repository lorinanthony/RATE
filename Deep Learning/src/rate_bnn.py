import numpy as np

def compute_B(X, H, M_W, V_W, b, C, effect_size_analogue="covariance"):
	"""
	Compute the means and covariance of the effect size analogues B for a Bayesian neural network
	described in (Ish-Horowicz et al., 2019)

	Args:
		X: array of inputs with shape (n_examples, n_input_dimensions)
		H: array of penultimate network layer outputs with shape (n_examples, final_hidden_layer_size)
		M_W: Array of final weight matrix means with shape (final_hidden_layer_size, n_classes)
		V_W: Array of final weight matrix variances with shape (final_hidden_layer_size, n_classes)
		b: Final layer (deterministic) bias
		C: number of classes
		effect_size_analogue: Projection operator for computing effect size analogues. Either "linear" (pseudoinverse) or "covariance" (default).

	Returns:
		M_B: an array of means of B, the multivariate effect size analgoues, with shape (n_classes, n_pixels)
		V_B: an array of covariance of B, the multivariate effect size analgoues, with shape (n_classes, n_pixels, n_pixels)

	"""
	assert X.shape[0]==H.shape[0], "Number of examples (first dimension) of X and H must match"
	assert b.shape[0]==C, "Number of bias units must match number of nodes in the logit layer"
	assert M_W.shape[1]==C, "Second dimension of logit weight means must match number of classes"
	assert V_W.shape[1]==C, "Second dimension of logit weight variances must match number of classes"
	assert M_W.shape[0]==V_W.shape[0], "means and variances of logit weight matrix must have the same shape"
	assert H.shape[1]==M_W.shape[0], "Second dimension of logit weight means must match H.shape[1], the penultimate layer size"

	n = H.shape[0]

	# Logits
	M_F = np.matmul(H, M_W) + b[np.newaxis,:]
	V_F = np.array([np.matmul(H*V_W[:,c], H.T) for c in range(C)])

	# Effect size analogues
	if effect_size_analogue == 'covariance':
		# Centred logits
		M_F_c = M_F - M_F.mean(axis=0)[np.newaxis,:]
		V_F_c = V_F # NB ignoring the additional variance due to centering + 1.0/n**2.0 * V_F.mean(axis=0)
		X_c = X - X.mean(axis=0)[np.newaxis,:]
		M_B = 1.0/(n-1.0) * np.array([np.matmul(X_c.T, M_F_c[:,c]) for c in range(C)])
		V_B = 1.0/(n-1.0)**2.0 * np.array([np.matmul(np.matmul(X_c.T, V_F_c[c,:,:]), X_c)for c in range(C)])
	elif effect_size_analogue == 'linear': 
		GenInv = np.linalg.pinv(X)
		# GenInv = np.matmul(np.linalg.pinv(np.matmul(X.T, X)), X.T)
		M_B_mat = np.matmul(GenInv, M_F)
		M_B = np.array([M_B_mat[:, c] for c in range(C)])
		V_B = np.array([np.matmul(np.matmul(GenInv, V_F[c, :, :]), GenInv.T) for c in range(C)])
	else: 
		print "Using covariance effect size analogue...\n"
		M_B = 1.0/(n-1.0) * np.array([np.matmul(X_c.T, M_F_c[:,c]) for c in range(C)])
		V_B = 1.0/(n-1.0)**2.0 * np.array([np.matmul(np.matmul(X_c.T, V_F_c[c,:,:]), X_c)for c in range(C)])
	    
	return M_B, V_B
    
def RATE(mu_c, Lambda_c, nullify=None):
    """
	Compute RATE values fromt means and covariances of the effect
	size analgoues of a single class.

	Args:
		mu_c: Array of means of the effect size analgoues with shape (n_variables,).
		Lambda_c: Array of covariances of the effect size analogues with shape (n_variables, n_variables)
		nullify: Array of indices to be ignored (default None means include all variables).

	Returns:
		Array of RATE values with shape (n_variables,)
    """

    mu = mu_c
    Lambda = Lambda_c
    
    J = np.arange(Lambda.shape[0])
    if nullify is not None:
        J = np.delete(J, nullify, axis=0)

    print "Computing RATE with {} variables".format(J.shape[0])
    print "Variable #:",
    
    def single_marker_kld(j):
        
        if j%100==0:
            print j, 
        
        if nullify is not None:
            j = np.array(np.unique(np.concatenate(([j], nullify)), axis=0))
        m = mu[j]
        Lambda_red = np.delete(Lambda, j, axis=0)[:,j]
        alpha = np.matmul(Lambda_red.T,
                          np.linalg.solve(np.delete(np.delete(Lambda, j, axis=0), j, axis=1),
                                          Lambda_red))
        if nullify is None:
            return 0.5 * m**2.0 * alpha
        else:
            return 0.5 * np.matmul(np.matmul(m.T, alpha), m)
    
    KLD = [single_marker_kld(j) for j in J]
    KLD = KLD / np.sum(KLD)
    print "done"
    return KLD

def RATE_BNN(X, H, M_W, V_W, b, C, effect_size_analogue="covariance"):
	"""
	Compute RATE values for the Bayesian neural network described in (Ish-Horowicz et al., 2019).
	If C>2 there is one RATE value per pixel per class. Note that a binary classificaiton task
	uses C=1 as there is a single output node in the network.

	This function wraps compute_B (which computes effect size analgoues) and RATE.

	Args:
		X: array of inputs with shape (n_examples, n_input_dimensions)
		H: array of penultimate network layer outputs with shape (n_examples, final_hidden_layer_size)
		M_W: Array of final weight matrix means with shape (final_hidden_layer_size, n_classes)
		V_W: Array of final weight matrix variances with shape (final_hidden_layer_size, n_classes)
		b: Final layer (deterministic) bias
		C: number of classes (output dimensionality)
		effect_size_analogue: Projection operator for computing effect size analogues. Either "linear" (pseudoinverse) or "covariance" (default).
		
	Returns:
		List of array of RATE values with shaplength C. Each array in the list has shape (n_variables,)
	"""

	M_B, V_B = compute_B(X, H, M_W, V_W, b, C, effect_size_analogue)
	try:
		if C > 2:
			return [RATE(mu_c=M_B[c,:],Lambda_c=V_B[c,:,:]) for c in range(C)]
		else:
			return RATE(mu_c=M_B[0,:],Lambda_c=V_B[0,:,:])
	except np.linalg.LinAlgError as err:
		if 'Singular matrix' in str(err):
			print "Computing RATE led to singlar matrices. Try using the nullify argument to ignore uninformative variables."
			print err
			return None
		else:
			raise err