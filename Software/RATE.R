### Predictor Prioritization via RelATive cEntrality (RATE) centrality measures ###

### NOTE: This function assumes that one has already obtained (posterior) draws/estimates of a nonparametric or nonlinear function as suggested in Crawford et al. (2018) ###

### Review of the function parameters ###
#'X' is the nxp design matrix (e.g. genotypes) where n is the number of samples and p is the number of dimensions. This is the original input data
#'f.draws' is the Bxn matrix of the nonparametric model estimates (i.e. f.hat) with B being the number of sampled (posterior) draws;
#'nullify' is an optional vector specifying a given predictor variable effect that user wishes to "remove" from the data set. An example of this corresponds to Figures 2(b)-(d) in Crawford et al. (2018);
#'snp.nms' is an optional vector specifying the names of the predictor variables;
#'cores' is a parameter detailing the number of cores to parallelize over. If too many are assigned, RATE will set this variable to maximum number of cores that are available on the operating machine.

#NOTE: As decribed in the Supplementary Material of Crawford et al. (2018), we implement different matrix factorizations
#for two cases: (i) n > p, and (ii) n < p. Again, n is the number of samples and p is the number of predictors.

######################################################################################
######################################################################################
######################################################################################

RATE = function(X = X, f.draws = f.draws, nullify = NULL, snp.nms = snp.nms, cores = 1){
  
  ### Install the necessary libraries ###
  usePackage("doParallel")
  usePackage("MASS")
  usePackage("Matrix")
  
  ### Determine the number of Cores for Parallelization ###
  if(cores > 1){
    if(cores>detectCores()){warning("The number of cores you're setting is larger than detected cores!");cores = detectCores()}
  }
  
  ### First Run the Matrix Factorizations ###  
  if(nrow(X) < ncol(X)){
    #In case X has linearly dependent columns, first take SVD of X to get v.
    svd_X = svd(X)
    r_X = sum(svd_X$d>1e-10)
    u = with(svd_X,(1/d[1:r_X]*t(u[,1:r_X])))
    v = svd_X$v[,1:r_X]
    
    # Now, calculate Sigma_star
    SigmaFhat = cov(f.draws)
    Sigma_star = u %*% SigmaFhat %*% t(u)
    
    # Now, calculate U st Lambda = U %*% t(U)
    svd_Sigma_star = svd(Sigma_star)
    r = sum(svd_Sigma_star$d > 1e-10)
    U = t(ginv(v)) %*% with(svd_Sigma_star, t(1/sqrt(d[1:r])*t(u[,1:r])))
  }else{
    beta.draws = t(ginv(X)%*%t(f.draws))
    V = cov(beta.draws); #V = as.matrix(nearPD(V)$mat)
    D = ginv(V)
    svd_D = svd(D)
    r = sum(svd_D$d>1e-10)
    U = with(svd_D,t(sqrt(d[1:r])*t(u[,1:r])))
  }
  Lambda = tcrossprod(U)
  
  ### Compute the Kullback-Leibler divergence (KLD) for Each Predictor ###
  mu = c(ginv(X)%*%colMeans(f.draws))
  int = 1:length(mu); l = nullify;
  
  if(length(l)>0){int = int[-l]}
  
  if(nrow(X) < ncol(X)){
    KLD = foreach(j = int)%dopar%{
      q = unique(c(j,l))
      m = mu[q]
      U_Lambda_sub = qr.solve(U[-q,],Lambda[-q,q,drop=FALSE])
      kld = crossprod(U_Lambda_sub%*%m)/2
      names(kld) = snp.nms[j]
      kld
    }
  }else{
    KLD = foreach(j = int)%dopar%{
      q = unique(c(j,l))
      m = mu[q]
      alpha = t(Lambda[-q,q])%*%ginv(as.matrix(nearPD(Lambda[-q,-q])$mat))%*%Lambda[-q,q]
      kld = (t(m)%*%alpha%*%m)/2
      names(kld) = snp.nms[j]
      kld
    }
  }
  KLD = unlist(KLD)
  
  ### Compute the corresponding “RelATive cEntrality” (RATE) measure ###
  RATE = KLD/sum(KLD)
  
  ### Find the entropic deviation from a uniform distribution ###
  Delta = sum(RATE*log((length(mu)-length(nullify))*RATE))
  
  ### Calibrate Delta via the effective sample size (ESS) measures from importance sampling ###
  #(Gruber and West, 2016, 2017)
  ESS = 1/(1+Delta)*100
  
  ### Return a list of the values and results ###
  return(list("KLD"=KLD,"RATE"=RATE,"Delta"=Delta,"ESS"=ESS))
}

### Define the Package ###
usePackage <- function(p) {
  if (!is.element(p, installed.packages()[,1]))
    install.packages(p, dep = TRUE)
  require(p, character.only = TRUE)
}