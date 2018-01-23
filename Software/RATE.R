### Predictor Prioritization via RelATive cEntrality (RATE) centrality measures ###

### NOTE: This function assumes that one has already obtained (posterior) mean, covariance, or other estimation quantities as suggested in Crawford et al. (2018) ###

### Review of the function parameters ###
#'mu' is the px1 vector of the effect sizes posterior mean;
#'Sigma' is the pxp positive semi-definite covariance matrix between predictor variables (should be a matrix and not a data frame);
#'Lambda' is the corresponding pxp precision matrix, derived as Lambda = Sigma^{-1} (again, should be a matrix and not a data frame); 
#'nullify' is an optional vector specifying a given predictor variable effect that user wishes to "remove" from the data set. An example of this corresponds to Figures 2(b)-(d) in Crawford et al. (2018);
#'snp.nms' is an optional vector specifying the names of the predictor variables;
#'cores' is a parameter detailing the number of cores to parallelize over. If too many are assigned, RATE will set this variable to maximum number of cores that are available on the operating machine.

######################################################################################
######################################################################################
######################################################################################

RATE = function(mu = mu, Sigma = Sigma, Lambda = Lambda, nullify = NULL, snp.nms = snp.nms, cores = 1){
  
  ### Install the necessary libraries ###
  usePackage("doParallel")
  usePackage("MASS")
  
  ### Determine the number of Cores for Parallelization ###
  if(cores > 1){
    if(cores>detectCores()){warning("The number of cores you're setting is larger than detected cores!");cores = detectCores()}
  }
  
  ### Compute the Kullback-Leibler divergence (KLD) for each predictor ###
  int = 1:length(mu); l = nullify;
  
  if(length(l)>0){int = int[-l]}
  
  KLD = foreach(j = int)%dopar%{
    q = unique(c(l,j))
    m = mu[q]
    alpha = t(Lambda[-q,q])%*%ginv(Lambda[-q,-q])%*%Lambda[-q,q]
    kld = (-log(det(Sigma[-q,-q]%*%Lambda[-q,-q]))+sum(Sigma[-q,-q]*Lambda[-q,-q])+length(q)-length(mu)+t(m)%*%alpha%*%m)/2
    names(kld) = snp.nms[j]
    kld
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