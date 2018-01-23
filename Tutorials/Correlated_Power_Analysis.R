#NOTE: This script will demonstrate the power of distributional centrality via RATE measures. Specifically it compares:
#(1) L1- regularized lasso regression; 
#(2) L2-regularized ridge regression;
#(3) The combined regularization utilized by the elastic net (Waldmann et al., 2013); 
#(4) A commonly used spike and slab prior model, also known as Bayesian variable selection regression (e.g. Guan and Stephens, 2011), which computes posterior inclusion probabilities (PIPs) for each covariate as a mixture of a point mass at zero and a diffuse normal centered around zero

#NOTE: This script is based on a simple (and small) genetics example where we simulate 
#genotype data for n = 100 individuals with p = 250 correlated genetic variants. We randomly 
#assume that p∗ = 30 are causal and have true association with #the generated (continuous)
#phenotype y. We then assume that the p* predictor variables explain a fixed Vx = 50%
#(phenotypic variance explained; PVE) of the total variance in the response V(y). This
#parameter Vx can alternatively be described as a factor controlling the signal-to-noise ratio.The 
#parameter rho represents the proportion of Vx that is contributed by additive effects versus
#interaction effects. Namely, the additive effects make up rho%, while the pairwise interactions 
#make up the remaining (1 − rho)%.

### Clear Console ###
cat("\014")

### Clear Environment ###
rm(list = ls(all = TRUE))

### Load in the R libraries ###
library(adegenet)
library(corpcor)
library(doParallel)
library(glmnet)
library(MASS)
library(Matrix)
library(mvtnorm)
library(Rcpp)
library(RcppArmadillo)
library(RcppParallel)
library(varbvs)

### Load in the RATE R functions ###
source("~/Dropbox/KLD Project/GitHub/Software/RATE.R")

### Load in the C++ BAKR functions ###
sourceCpp("~/Dropbox/KLD Project/Code/BAKRGibbs.cpp")

### Define the Compute Power Function ###
compute.power <- function(pvals,SNPs){
  nsnps = length(pvals)
  Pos = SNPs #True Positive Set
  Negs = names(pvals)[which(names(pvals)%in%SNPs==FALSE)] #True Negative Set
  x = foreach(i = 1:nsnps)%dopar%{
    v = sort(pvals,decreasing = TRUE)[1:i] #Test Positives
    z = pvals[which(names(pvals)%in%names(v)==FALSE)] #Test Negatives
    
    TP = length(which(names(v)%in%Pos==TRUE))
    FP = length(which(names(v)%in%Pos==FALSE))
    TN = length(which(names(z)%in%Negs==TRUE))
    FN = length(which(names(z)%in%Negs==FALSE))
    
    TPR = TP/(TP+FN); FPR = FP/(FP+TN); FDR = FP/(FP+TP)
    c(TPR,FPR,FDR)
  }
  return(matrix(unlist(x),ncol = 3,byrow = TRUE))
}

######################################################################################
######################################################################################
######################################################################################

### Set the random seed to reproduce research ###
set.seed(11151990)

### Set up simulation parameters ###
n = 100; p = 250; pve=0.5; rho=0.75;

### The Number of Causal Variables ###
ncausal = 30 
ncausal1= 10 #Set 1 of causal SNPs 
ncausal2 = 10 #Set 2 of Causal SNPs
ncausal3 = ncausal-(ncausal1+ncausal2) #Set 3 of Causal SNPs with only marginal effects

### Generate the data ###
X = glSim(n,p-ncausal,ncausal,parallel = TRUE, LD = TRUE,ploidy = 2)
X  = as.matrix(X)-1+1
#Xmean=apply(X, 2, mean); Xsd=apply(X, 2, sd); Geno=t((t(X)-Xmean)/Xsd)
colnames(X) = paste("SNP",1:ncol(X),sep="")

### Select Causal SNPs ###
s=(p-ncausal+1):p
s1=sample(s, ncausal1, replace=F)
s2=sample(s[s%in%s1==FALSE], ncausal2, replace=F)
s3=sample(s[s%in%c(s1,s2)==FALSE], ncausal3, replace=F)

### Generate the Marginal Effects ###
Xmarginal=X[,s]
beta=runif(dim(Xmarginal)[2])
y_marginal=c(Xmarginal%*%beta)
beta=beta*sqrt(pve*rho/var(y_marginal))
y_marginal=Xmarginal%*%beta

### Generate the Pairwise Interaction Matrix W ###
Xcausal1=X[,s1]; Xcausal2=X[,s2];
W=c()
for(i in 1:ncausal1){
  W=cbind(W,Xcausal1[,i]*Xcausal2)
}
dim(W)

### Generate the Epistatic Effects ###
gamma=runif(dim(W)[2])
y_epi=c(W%*%gamma)
gamma=gamma*sqrt(pve*(1-rho)/var(y_epi))
y_epi=W%*%gamma

### Generate the Random Error Terms ###
y_err=rnorm(n)
y_err=y_err*sqrt((1-pve)/var(y_err))

### Generate the Phenotypes ###
y=c(y_marginal+y_epi+y_err)
y=(y-mean(y))/(sd(y))

######################################################################################
######################################################################################
######################################################################################

### Running the Bayesian Gaussian Process (GP) Regression Model ###

### Create the Nonlinear Covariance Matrix with the Gaussian Kernel ###
#This function takes on two arguments:
#(1) The Genotype matrix X. This matrix should be fed in as a pxn matrix. That is, predictor
#are the rows and subjects/patients/cell lines are the columns.
#(2) The bandwidth (also known as a smoothing parameter or lengthscale) h. For example, the 
#Gaussian kernel can be specified as k(u,v) = exp{||u−v||^2/2h^2}.

h = median(as.matrix(dist(X)))
Kn = GaussKernel(t(X),1/(h^2*2)); diag(Kn)=1 # 

### Set up the desired number of posterior draws ###
mcmc.iter = 1e4

### Fit the GP Regression ###
fhat = Kn %*% solve(Kn + diag(n), y) #Posterior mean
fhat.rep = rmvnorm(mcmc.iter,fhat,Kn - Kn %*% solve(Kn+ diag(n),Kn))

### Get the Posterior Draws of the Effect Size Analogue ###
beta.tilde.draws = t(ginv(X) %*% t(fhat.rep)) #Posterior Draws

#NOTE: We formally define the effect size analogue as the result of projecting the design 
#matrix X onto the nonlinear response vector f, where beta = Proj(X,f) = X^+f with X^+ 
#symbolizing the Moore-Penrose generalized inverse.

### Find the Approximate Posterior Mean, Covariance, and Precision ###
mu = colMeans(beta.tilde.draws)
Sigma = cov(beta.tilde.draws); Sigma = as.matrix(nearPD(Sigma)$mat)
Lambda = ginv(Sigma)

### NOTE: Sometimes it is worth using the shrinkage version of the covariance matrix V = cov.shrink(V). This can help stabilize the KLD and RATE computations ### 

### Compute the First Order Centrality of each Predictor Variable ###
cores = cores=detectCores()

### Run the RATE Function ###
nl = NULL
res = RATE(mu,Sigma,Lambda,snp.nms = colnames(X),cores = cores)

#The function results in a list with: 
#(1) The raw Kullback-Leibler divergence measures (RATE$KLD); 
#(2) The relative centrality measures (RATE$RATE); 
#(3) The entropic deviance from uniformity (RATE$Detla); and 
#(4) The calibrating approximate effect sample size (ESS) measures from importance sampling (Gruber and West, 2016, 2017)

### Get the Results ###
rates = res$RATE

######################################################################################
######################################################################################
######################################################################################

### Bayes Variable Selection ###
fit = varbvs(X,Z=NULL,y)
pips = fit$pip

### LASSO ###
fit= cv.glmnet(X, y,intercept=FALSE,alpha=0.95)
lasso = as.matrix(coef(fit,s = fit$lambda.1se))
lasso = c(lasso[-1,])

### Elastic Net ###
fit= cv.glmnet(X, y,intercept=FALSE,alpha=0.25)
enet = as.matrix(coef(fit,s = fit$lambda.1se))
enet = c(enet[-1,])

### Ridge ###
fit= cv.glmnet(X, y,intercept=FALSE,alpha=0)
ridge = as.matrix(coef(fit,s = fit$lambda.1se))
ridge = c(ridge[-1,])

######################################################################################
######################################################################################
######################################################################################

### Compute the Power of Each Metric ###
b = rates; names(b) = colnames(X)
power.rate = compute.power(b,colnames(X)[s])

b = abs(ridge); names(b) = colnames(X)
power.ridge = compute.power(b,colnames(X)[s])

b = pips
power.pips= compute.power(b,colnames(X)[s])

b = abs(lasso); names(b) = colnames(X)
power.lasso= compute.power(b,colnames(X)[s])

b = abs(enet); names(b) = colnames(X)
power.enet= compute.power(b,colnames(X)[s])

### Visualize these power comparisons ###
plot(power.rate[,2],power.rate[,1],type = "l",lty = 1, lwd = 2, col = "blue", xlab = "False Positive Rate",ylab = "True Positive Rate", ylim = c(0,1),bty = "n")
lines(power.pips[,2],power.pips[,1],type = "l",lty = 3, lwd = 2, col = "magenta")
lines(power.lasso[,2],power.lasso[,1],type = "l",lty = 5, lwd = 2, col = "forest green")
lines(power.enet[,2],power.enet[,1],type = "l",lty = 6, lwd = 2, col = "dark red")
lines(power.ridge[,2],power.ridge[,1],type = "l",lty = 2, lwd = 2, col = "grey60")
abline(a=0,b=1)
legend("topleft",legend = c("RATE","PIPs","LASSO","Elastic Net","Ridge"), lty =c(1,4,5,6,2,3), lwd = c(2,2,2,2,2,3),col = c("blue","magenta","forest green","dark red","grey60"),ncol=2,bty = "n")
