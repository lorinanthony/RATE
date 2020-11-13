### Variable Prioritization via RelATive cEntrality (RATE) centrality measures ###

### NOTE: This function assumes that one has already obtained (posterior) draws/estimates of a nonparametric or nonlinear function as suggested in Crawford et al. (2018) ###

### Review of the function parameters ###
#'X' is the nxp design matrix (e.g. genotypes) where n is the number of samples and p is the number of dimensions. This is the original input data
#'f.draws' is the Bxn matrix of the nonparametric model estimates (i.e. f.hat) with B being the number of sampled (posterior) draws;
#'prop.var' is the desired proportion of variance that the user wants to explain when applying singular value decomposition (SVD) to the design matrix X (this is preset to 1);
#'low.rank' is a boolean variable detailing if the function will use low rank matrix approximations to compute the RATE values --- note that this highly recommended in the case that the number of covariates (e.g. SNPs, genetic markers) is large; 
#'rank.r' is a parameter detailing which matrix rank order to compute the RATE measures --- the default is min(n,p); however, this can be specified to a smaller value at the possible cost of accuracy and power;
#'nullify' is an optional vector specifying a given predictor variable effect that user wishes to "remove" from the data set;
#'snp.nms' is an optional vector specifying the names of the predictor variables;
#'cores' is a parameter detailing the number of cores to parallelize over. If too many are assigned, RATE will set this variable to maximum number of cores that are available on the operating machine.

######################################################################################
######################################################################################
######################################################################################

RATE = function(X = X, f.draws = f.draws,prop.var = 1, low.rank = FALSE, rank.r = min(nrow(X),ncol(X)), nullify = NULL,snp.nms = snp.nms, cores = 1){
  
  ### Install the necessary libraries ###
  usePackage("doParallel")
  usePackage("MASS")
  usePackage("Matrix")
  usePackage("svd")
  
  ### Determine the number of Cores for Parallelization ###
  if(cores > 1){
    if(cores>detectCores()){warning("The number of cores you're setting is larger than detected cores!");cores = detectCores()}
  }
  
  ### Register those Cores ###
  registerDoParallel(cores=cores)
  
  ### First Run the Matrix Factorizations ###  
  svd_X = propack.svd(X,rank.r); 
  dx = svd_X$d > 1e-10
  px = cumsum(svd_X$d^2/sum(svd_X$d^2)) < prop.var
  r_X = dx&px 
  u = with(svd_X,(1/d[r_X]*t(u[,r_X])))
  v = svd_X$v[,r_X]  
  
  if(low.rank==TRUE){
  # Now, calculate Sigma_star
  SigmaFhat = cov(f.draws)
  Sigma_star = u %*% SigmaFhat %*% t(u)
  
  # Now, calculate U st Lambda = U %*% t(U)
  svd_Sigma_star = propack.svd(Sigma_star,rank.r)
  r = svd_Sigma_star$d > 1e-10
  U = t(ginv(v)) %*% with(svd_Sigma_star, t(1/sqrt(d[r])*t(u[,r])))
  
  mu = v%*%u%*%colMeans(f.draws)
  }else{
    beta.draws = t(ginv(X)%*%t(f.draws))
    V = cov(beta.draws); #V = as.matrix(nearPD(V)$mat)
    D = ginv(V)
    svd_D = svd(D)
    r = sum(svd_D$d>1e-10)
    U = with(svd_D,t(sqrt(d[1:r])*t(u[,1:r])))
    
    mu = colMeans(beta.draws)
  }
  
  ### Create Lambda ###
  Lambda = tcrossprod(U)
  
  ### Compute the Kullback-Leibler divergence (KLD) for Each Predictor ###
  int = 1:length(mu); l = nullify;
  
  if(length(l)>0){int = int[-l]}
  
  KLD = foreach(j = int, .combine='c')%dopar%{
    q = unique(c(j,l))
    m = abs(mu[q])
  
    U_Lambda_sub = sherman_r(Lambda,V[,q],V[,q])
    #U_Lambda_sub = U_Lambda_sub[-q,-q]
    alpha = t(U_Lambda_sub[-q,q])%*%U_Lambda_sub[-q,-q]%*%U_Lambda_sub[-q,q]
    kld = kld = (t(m)%*%alpha%*%m)/2
    names(kld) = snp.nms[j]
    kld
  }
  # 
  # if(nrow(X) < ncol(X)){
  #   KLD = foreach(j = int, .combine='c')%dopar%{
  #     q = unique(c(j,l))
  #     m = abs(mu[q])
  #     
  #     U_Lambda_sub = qr.solve(U[-q,],Lambda[-q,q,drop=FALSE])
  #     kld = crossprod(U_Lambda_sub%*%m)/2
  #     names(kld) = snp.nms[j]
  #     kld
  #   }
  # }else{
  #   KLD = foreach(j = int, .combine='c')%dopar%{
  #     q = unique(c(j,l))
  #     m = mu[q]
  #     alpha = t(Lambda[-q,q])%*%ginv(as.matrix(nearPD(Lambda[-q,-q])$mat))%*%Lambda[-q,q]
  #     kld = (t(m)%*%alpha%*%m)/2
  #     names(kld) = snp.nms[j]
  #     kld
  #   }
  # }
  # 
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

sherman_r <- function(Ap, u, v) {
  Ap - (Ap %*% u %*% t(v) %*% Ap)/drop(1 + u %*% t(v) %*% Ap)
}

### Define the Package ###
usePackage <- function(p) {
  if (!is.element(p, installed.packages()[,1]))
    install.packages(p, dep = TRUE)
  require(p, character.only = TRUE)
}
