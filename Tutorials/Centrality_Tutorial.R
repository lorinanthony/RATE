#NOTE: This script will demonstrate orders of distributional centrality via RATE measures. Specifically it shows:
#(1) How to compute a covariance matrix using the Gaussian kernel function;
#(2) How to fit a standard Bayesian Gaussian process (GP) regression model;
#(3) Retrieve effect size analogue estimates for the original predictor variables;
#(4) Prioritize variables via their first, second, third, and fourth order centrality.

#NOTE: This script is based on a simple (and small) genetics example where we simulate 
#genotype data for n = 500 individuals with p = 25 measured genetic variants. We randomly 
#assume the last three predictors p∗ = {23, 24, 25} are causal and have true association with #the generated (continuous) phenotype y. We then assume that the p* predictor variables 
#explain a fixed Vx = 75% (phenotypic variance explained; PVE) of the total variance in the 
#response V(y). This parameter Vx can alternatively be described as a factor controlling the 
#signal-to-noise ratio. The parameter rho represents the proportion of Vx that is contributed by additive effects versus interaction effects. Namely, the additive effects make up rho%, while the pairwise interactions make up the remaining (1 − rho)%.

### Clear Console ###
cat("\014")

### Clear Environment ###
rm(list = ls(all = TRUE))

### Load in the R libraries ###
library(adegenet)
library(corpcor)
library(doParallel)
library(MASS)
library(Matrix)
library(mvtnorm)
library(Rcpp)
library(RcppArmadillo)
library(RcppParallel)

### Load in the RATE R functions ###
source("~/RATE.R")

### Load in the C++ BAKR functions ###
sourceCpp("~/BAKRGibbs.cpp")

######################################################################################
######################################################################################
######################################################################################

### Set the random seed to reproduce research ###
set.seed(11151990)

### Set up simulation parameters ###
n = 500; p = 25; pve=0.75; rho=0.5;

### Generate the data ###
maf <- 0.05 + 0.45*runif(p)
X   <- (runif(n*p) < maf) + (runif(n*p) < maf)
X   <- matrix(as.double(X),n,p,byrow = TRUE)
#Xmean=apply(X, 2, mean); Xsd=apply(X, 2, sd); Geno=t((t(X)-Xmean)/Xsd)
colnames(X) = paste("SNP",1:ncol(X),sep="")

### Set the number of causal variables ###
ncausal = 3
s=c(23:25)

### Generate the Marginal Effects ###
Xmarginal=X[,s]
beta1=rep(1,ncausal)
y_marginal=c(Xmarginal%*%beta1)
beta1=beta1*sqrt(pve*rho/var(y_marginal))
y_marginal=Xmarginal%*%beta1

### Generate the Epistatic Effects ###
Xepi=cbind(X[,s[1]]*X[,s[3]],X[,s[2]]*X[,s[3]])
beta2=c(1,1)
y_epi=c(Xepi%*%beta2)
beta2=beta2*sqrt(pve*(1-rho)/var(y_epi))
y_epi=Xepi%*%beta2

### Generate the Random Error Terms ###
y_err=rnorm(n)
y_err=y_err*sqrt((1-pve)/var(y_err))

### Generate the Phenotypes ###
y=c(y_marginal+y_epi+y_err)

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
Sigma = cov(beta.tilde.draws)
Lambda = ginv(Sigma)

### NOTE: Sometimes it is worth using the shrinkage version of the covariance matrix V = cov.shrink(V). This can help stabilize the KLD and RATE computations ### 

######################################################################################
######################################################################################
######################################################################################

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
DELTA = res$Delta
ESS = res$ESS

### Plot the results with the uniformity line ###
barplot(rates,xlab = "Covariates",ylab=expression(RATE(tilde(beta)[j])),names.arg ="",col = ifelse(c(1:length(mu))%in%s,"blue","grey80"),border=NA,cex.names = 0.6,ylim=c(0,0.6),cex.lab=1.25,cex.axis = 1.25)
lines(x = -0.5:30.5,y = rep(1/(p-length(nl)),32),col = "red",lty=2,lwd=2)
legend("topleft",legend=c(as.expression(bquote(DELTA~"="~.(round(DELTA,3)))),as.expression(bquote("ESS ="~.(round(ESS,2))*"%"))),bty = "n",pch = 20,col = "red")

######################################################################################
######################################################################################
######################################################################################

### Find Second Order Centrality by Nullifying the most Associated Predictor Variable ###

### Run the RATE Function ###
nl = which(res$KLD%in%sort(res$KLD,decreasing=TRUE)[1])
res2 = RATE(mu,Sigma,Lambda,nullify = nl,snp.nms = colnames(X),cores = cores)

### Get the Results ###
rates = res2$RATE
DELTA = res2$Delta
ESS = res2$ESS

### Plot the results with the uniformity line ###
barplot(rates,xlab = "Covariates",ylab=bquote(RATE(tilde(beta)[j]~"|"~tilde(beta)[.(as.integer(nl))]=="0")),names.arg = "",col = ifelse(c(1:length(mu))[-nl]%in%s,"blue","grey80"),border=NA,cex.names = 0.6,ylim=c(0,0.6),cex.lab=1.25,cex.axis = 1.25)
lines(x = -0.5:29.5,y = rep(1/(p-length(nl)),31),col = "red",lty=2,lwd=2)
legend("topleft",legend=c(as.expression(bquote(DELTA~"="~.(round(DELTA,3)))),as.expression(bquote("ESS ="~.(round(ESS,2))*"%"))),bty = "n",pch = 20,col = "red")

######################################################################################
######################################################################################
######################################################################################

### Find Third Order Centrality by Nullifying the most Associated Predictor Variable ###

### Run the RATE Function ###
nl = c(nl,which(res2$KLD%in%sort(res2$KLD,decreasing=TRUE)[1]))
res3 = RATE(mu,Sigma,Lambda,nullify = nl,snp.nms = colnames(X),cores = cores)

### Get the Results ###
rates = res3$RATE
DELTA = res3$Delta
ESS = res3$ESS

### Plot the results with the uniformity line ###
barplot(rates,xlab = "Covariates",ylab=bquote(RATE(tilde(beta)[j]~"|"~tilde(beta)[.(as.integer(nl))]=="0")),names.arg = "",col = ifelse(c(1:length(mu))[-nl]%in%s,"blue","grey80"),border=NA,cex.names = 0.6,ylim=c(0,0.6),cex.lab=1.25,cex.axis = 1.25)
lines(x = -0.5:28.5,y = rep(1/(p-length(nl)),30),col = "red",lty=2,lwd=2)
legend("topleft",legend=c(as.expression(bquote(DELTA~"="~.(round(DELTA,3)))),as.expression(bquote("ESS ="~.(round(ESS,2))*"%"))),bty = "n",pch = 20,col = "red")

######################################################################################
######################################################################################
######################################################################################

### Find Fourth Order Centrality by Nullifying the most Associated Predictor Variable ###

### Run the RATE Function ###
nl = c(nl,which(res3$KLD%in%sort(res3$KLD,decreasing=TRUE)[1]))
res4 = RATE(mu,Sigma,Lambda,nullify = nl,snp.nms = colnames(X),cores = cores)

### Get the Results ###
rates = res4$RATE
DELTA = res4$Delta
ESS = res4$ESS

### Plot the results with the uniformity line ###
barplot(rates,xlab = "Covariates",ylab=bquote(RATE(tilde(beta)[j]~"|"~tilde(beta)[.(as.integer(nl))]=="0")),names.arg = "",col = ifelse(c(1:length(mu))[-nl]%in%s,"blue","grey80"),border=NA,cex.names = 0.6,ylim=c(0,0.6),cex.lab=1.25,cex.axis = 1.25)
lines(x = -0.5:27.5,y = rep(1/(p-length(nl)),29),col = "red",lty=2,lwd=2)
legend("topleft",legend=c(as.expression(bquote(DELTA~"="~.(round(DELTA,3)))),as.expression(bquote("ESS ="~.(round(ESS,2))*"%"))),bty = "n",pch = 20,col = "red")
