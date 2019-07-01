import numpy as np
import multiprocessing as mp
import time
import warnings

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
		raise ValueError("Unrecognised effect_size_analogue {}, please use `covariance` or `linear`".format(effect_size_analogue))
	    
	return M_B, V_B
    
def RATE_sequential(mu_c, Lambda_c, nullify=None):
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

    print("Computing RATE with {} variables".format(J.shape[0]))
    print("Variable #:", end=" ")
    
    def single_marker_kld(j):
        
        if j%100==0:
            print(j, end=" ")
        
        if nullify is not None:
            j = np.array(np.unique(np.concatenate(([j], nullify)), axis=0))
        m = mu[j]
        Lambda_red = np.delete(Lambda, j, axis=0)[:,j]
        
        # in def'n of alpha below, 
        # changing np.linalg.solve to np.linalg.lstsq with rcond=None, to avoid singularity error 
        # when simulating data in Power Simulation Response to ICML Feedback.ipynb
        # (dana)
        alpha = np.matmul(Lambda_red.T, 
                          np.linalg.lstsq(np.delete(np.delete(Lambda, j, axis=0), j, axis=1),
                                          Lambda_red, rcond=None)[0])
        if nullify is None:
            return 0.5 * m**2.0 * alpha
        else:
            return 0.5 * np.matmul(np.matmul(m.T, alpha), m)
    
    KLD = [single_marker_kld(j) for j in J]
    print("done")
    return KLD / np.sum(KLD)

def groupRATE_sequential(mu_c, Lambda_c, groups, nullify=None):
	"""
	Group RATE

	Args:
		groups: List of lists, where groups[i] contains the indices of the variables in group i
	"""
	mu = mu_c
	Lambda = Lambda_c
    
	J = np.arange(Lambda.shape[0])
	if nullify is not None:
		J = np.delete(J, nullify, axis=0)

	print("Computing RATE with {} groups".format(len(groups)))
	print("Group #:", end=" ")

	def group_kld(group, idx):
		if idx%100 == 0:
			print(idx, end=", ")

		if nullify is not None:
			j = np.array(np.unique(np.concatenate((group, nullify)), axis=0))
		else:
			j = group
		m = mu[j]
		Lambda_red = np.delete(Lambda, j, axis=0)[:,j]

		# in def'n of alpha below, 
		# changing np.linalg.solve to np.linalg.lstsq with rcond=None, to avoid singularity error 
		# when simulating data in Power Simulation Response to ICML Feedback.ipynb
		# (dana)
		alpha = np.matmul(Lambda_red.T, 
							np.linalg.lstsq(np.delete(np.delete(Lambda, j, axis=0), j, axis=1),
											Lambda_red, rcond=None)[0])
		if nullify is None:
			return 0.5 * m**2.0 * alpha
		else:
			return 0.5 * np.matmul(np.matmul(m.T, alpha), m)

	KLD = [group_kld(group, idx) for idx, group in enumerate(groups)]
	print("done")
	return KLD / np.sum(KLD)

# Worker initialisation and function for groupRATE
var_dict = {}

def init_worker(mu, Lambda, p):
	var_dict["mu"] = mu
	var_dict["Lambda"] = Lambda
	var_dict["p"] = p

def worker_func(worker_args):
	"""
	Returns KLD
	"""
	j, idx, filepath = worker_args
	Lambda_np = np.frombuffer(var_dict["Lambda"]).reshape(var_dict["p"], var_dict["p"])
	mu_np = np.frombuffer(var_dict["mu"])
	m = mu_np[j]
	Lambda_red = np.delete(Lambda_np, j, axis=0)[:,j]

	alpha = np.matmul(Lambda_red.T, 
					  np.linalg.lstsq(np.delete(np.delete(Lambda_np, j, axis=0), j, axis=1),
									  Lambda_red, rcond=None)[0])

	if isinstance(m, float):
		out = 0.5 * m**2.0 * alpha
	else:
		out = 0.5 * np.matmul(np.matmul(m.T, alpha), m)
	
	if filepath is not None:
		with open(filepath + "kld_{}.csv".format(idx), "w") as f:
			f.write(str(out))

	return out

def RATE(mu_c, Lambda_c, nullify=None, n_workers=1, filepath=None):
	if nullify is not None and n_workers > 1:
		warnings.warn("Using nullify means a sequential RATE calculation")
		n_workers = 1
	
	if n_workers == 1:
		return RATE_sequential(mu_c, Lambda_c, nullify)
    
	p = mu_c.shape[0]
    
	print("Computing RATE for {} variables using {} worker(s)".format(p, n_workers))

	# Setup shared arrays
	mu_mp = mp.RawArray('d', p)    
	Lambda_mp = mp.RawArray('d', p*p)
	mu_np = np.frombuffer(mu_mp, dtype=np.float64)
	Lambda_np = np.frombuffer(Lambda_mp, dtype=np.float64).reshape(p, p)
	np.copyto(mu_np, mu_c)
	np.copyto(Lambda_np, Lambda_c)

    # Run pooled computation
	with mp.Pool(processes=n_workers, initializer=init_worker, initargs=(mu_c, Lambda_c, p)) as pool:
		result = np.array(pool.map(worker_func, [(j, j, filepath) for j in range(p)]))
	return result/result.sum()

def groupRATE(mu_c, Lambda_c, groups, nullify=None, n_workers=1, filepath=None):
    if nullify is not None and n_workers > 1:
        warnings.warn("Using nullify means a sequential groupRATE calculation")
        n_workers = 1

    if n_workers == 1:
        return groupRATE_sequential(mu_c, Lambda_c, groups, nullify)
    
    p = mu_c.shape[0]
    
    # Setup shared arrays
    mu_mp = mp.RawArray('d', p)    
    Lambda_mp = mp.RawArray('d', p*p)
    mu_np = np.frombuffer(mu_mp, dtype=np.float64)
    Lambda_np = np.frombuffer(Lambda_mp, dtype=np.float64).reshape(p, p)
    np.copyto(mu_np, mu_c)
    np.copyto(Lambda_np, Lambda_c)
        
    # Run pooled computation
    with mp.Pool(processes=n_workers, initializer=init_worker, initargs=(mu_c, Lambda_c, p)) as pool:
        result = np.array(pool.map(worker_func, [(group, idx, filepath) for idx, group in enumerate(groups)]))
    return result/result.sum()

def RATE_BNN(bnn, X, groups=None, nullify=None, effect_size_analogue="covariance",
	n_workers=1, return_esa_posterior=False, filepath=None):
	"""
	Compute RATE values for the Bayesian neural network described in (Ish-Horowicz et al., 2019).
	If C>2 there is one RATE value per pixel per class. Note that a binary classification task
	uses C=1 as there is a single output node in the network.

	This function wraps compute_B (which computes effect size analgoues) and RATE.

	Args:
		bnn: BNN object
		X: array of inputs with shape (n_examples, n_input_dimensions)
		groups: A list of lists where groups[i] is a list of indices of the variables in group i. Default is None (no group RATE)
		effect_size_analogue: Projection operator for computing effect size analogues. Either "linear" (pseudoinverse) or "covariance" (default).
		n_workers: number of workers for groupRATE
        return_esa_posterior: Controls whether the mean/covariance of the effect size analgoue posterior is also returned (default False)
        filepath: where to save the result of each worker (None means no saving and is the default)
		
	Returns:
		Tuple of (rate_vales, computation_time)
		If groups is None: rate_vales is a list of arrays of RATE values with length C. Each array in the list has shape (n_variables,).
		Otherwise, each array in rate_values has length len(groups).
	"""
	C = bnn.C
	M_W, V_W, b = bnn.var_params()
	H = bnn.compute_H(X)

	start_time = time.time()

	M_B, V_B = compute_B(X, H, M_W, V_W, b, C, effect_size_analogue)

	try:
		if C > 2:
			if groups is None:
				out = [RATE(mu_c=M_B[c,:],Lambda_c=V_B[c,:,:], nullify=nullify, n_workers=n_workers, filepath=filepath) for c in range(C)]
			else:
				out = [groupRATE(mu_c=M_B[c,:],Lambda_c=V_B[c,:,:], groups=groups, nullify=nullify, n_workers=n_workers, filepath=filepath) for c in range(C)]
			rate_time = time.time() - start_time
		else:
			if groups is None:
				out = RATE(mu_c=M_B[0,:],Lambda_c=V_B[0,:,:], nullify=nullify, n_workers=n_workers, filepath=filepath)
			else:
				out = groupRATE(mu_c=M_B[0,:],Lambda_c=V_B[0,:,:], groups=groups, nullify=nullify, n_workers=n_workers, filepath=filepath)
			rate_time = time.time() - start_time
	except np.linalg.LinAlgError as err:
		if 'Singular matrix' in str(err):
			print("Computing RATE led to singlar matrices. Try using the nullify argument to ignore uninformative variables.")
			print("Full exception:\n{}".format(err))
			return None, None
		else:
			raise err
            
	if return_esa_posterior:
		return out, rate_time, M_B, V_B
	else:
		return out, rate_time