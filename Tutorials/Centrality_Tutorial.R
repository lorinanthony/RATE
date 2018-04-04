#NOTE: This script will demonstrate orders of distributional centrality via RATE measures. Specifically it shows:
#(1) How to compute a covariance matrix using the Gaussian kernel function;
#(2) How to fit a standard Bayesian Gaussian process (GP) regression model;
#(3) Prioritize variables via their first, second, third, and fourth order distributional centrality.

#NOTE: This script is based on a simple (and small) genetics example where we simulate 
#genotype data for n individuals with p measured genetic variants. We randomly
#assume that three of the predictors p∗ = {23, 24, 25} are causal and have true association with 
#the generated (continuous) phenotype y. We then assume that the p* predictive markers 
#explain a fixed H2 % (phenotypic variance explained; PVE) of the total variance in the 
#response V(y). This parameter H2 can alternatively be described as a factor controlling the 
#signal-to-noise ratio. The parameter rho represents the proportion of H2 that is contributed by additive effects versus interaction effects. 
#Namely, the additive effects make up rho%, while the pairwise interactions (epistatic effects) make up the remaining (1 − rho)%.

######################################################################################
######################################################################################
######################################################################################

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
source("RATE.R")

### Load in the C++ BAKR functions ###
sourceCpp("BAKRGibbs.cpp")

######################################################################################
######################################################################################
######################################################################################

### Set the random seed to reproduce research ###
set.seed(11151990)

n = 250; p = 500; pve=0.75; rho=0.5;
ncausal1= 3 #Set 1 of causal SNPs

ncausal = ncausal1

maf <- 0.05 + 0.45*runif(p)
X   <- (runif(n*p) < maf) + (runif(n*p) < maf)
X   <- matrix(as.double(X),n,p,byrow = TRUE)
Xmean=apply(X, 2, mean); Xsd=apply(X, 2, sd); X=t((t(X)-Xmean)/Xsd)
s=c(23:25)

# Marginal Effects Only
Xmarginal=X[,s]
beta1=rep(1,ncausal) #0.5,0.5)
y_marginal=c(Xmarginal%*%beta1)
beta1=beta1*sqrt(pve*rho/var(y_marginal))
y_marginal=Xmarginal%*%beta1

#Pairwise Epistatic Effects
Xepi=cbind(X[,s[1]]*X[,s[3]],X[,s[2]]*X[,s[3]])
beta2=c(1,1)
y_epi=c(Xepi%*%beta2)
beta2=beta2*sqrt(pve*(1-rho)/var(y_epi))
y_epi=Xepi%*%beta2

# error
y_err=rnorm(n)
y_err=y_err*sqrt((1-pve)/var(y_err))

y=c(y_marginal+y_epi+y_err); #Full Model
colnames(X) = paste("SNP",1:ncol(X),sep="")

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

#NOTE: We formally define the effect size analogue as the result of projecting the design 
#matrix X onto the nonlinear response vector f, where beta = Proj(X,f) = X^+f with X^+ 
#symbolizing the Moore-Penrose generalized inverse.

######################################################################################
######################################################################################
######################################################################################

### Compute the First Order Centrality of each Predictor Variable ###
cores = cores=detectCores()

### Run the RATE Function ###
nl = NULL
res = RATE(X=X,f.draws=fhat.rep,snp.nms = colnames(X),cores = cores)

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
barplot(rates,xlab = "Covariates",ylab=expression(RATE(tilde(beta)[j])),names.arg ="",col = ifelse(c(1:p)%in%s,"blue","grey80"),border=NA,cex.names = 0.6,ylim=c(0,0.6),cex.lab=1.25,cex.axis = 1.25)
lines(x = -0.5:(p*1.5),y = rep(1/(p-length(nl)),p+1),col = "red",lty=2,lwd=2)
legend("topleft",legend=c(as.expression(bquote(DELTA~"="~.(round(DELTA,3)))),as.expression(bquote("ESS ="~.(round(ESS,2))*"%"))),bty = "n",pch = 20,col = "red")

######################################################################################
######################################################################################
######################################################################################

### Run the RATE Function ###
top = substring(names(res$KLD)[order(res$KLD,decreasing=TRUE)[1]],first = 4)
nl = c(nl,as.numeric(top))  
res2 = RATE(X=X,f.draws=fhat.rep,nullify = nl,snp.nms = colnames(X),cores = cores)

### Get the Results ###
rates = res2$RATE
DELTA = res2$Delta
ESS = res2$ESS

### Plot the results with the uniformity line ###
barplot(rates,xlab = "Covariates",ylab=bquote(RATE(tilde(beta)[j]~"|"~tilde(beta)[.(as.integer(nl))]=="0")),names.arg = "",col = ifelse(c(1:p)[-nl]%in%s,"blue","grey80"),border=NA,cex.names = 0.6,ylim=c(0,0.6),cex.lab=1.25,cex.axis = 1.25)
lines(x = -0.5:((p-1)*1.5),y = rep(1/(p-length(nl)),(p-1)+1),col = "red",lty=2,lwd=2)
legend("topleft",legend=c(as.expression(bquote(DELTA~"="~.(round(DELTA,3)))),as.expression(bquote("ESS ="~.(round(ESS,2))*"%"))),bty = "n",pch = 20,col = "red")

######################################################################################
######################################################################################
######################################################################################

### Find Third Order Centrality by Nullifying the most Associated Predictor Variable ###

### Run the RATE Function ###
top = substring(names(res2$KLD)[order(res2$KLD,decreasing=TRUE)[1]],first = 4)
nl = c(nl,as.numeric(top))
res3 = RATE(X=X,f.draws=fhat.rep,nullify = nl,snp.nms = colnames(X),cores = cores)

### Get the Results ###
rates = res3$RATE
DELTA = res3$Delta
ESS = res3$ESS

### Plot the results with the uniformity line ###
barplot(rates,xlab = "Covariates",ylab=bquote(RATE(tilde(beta)[j]~"|"~tilde(beta)[.(as.integer(nl))]=="0")),names.arg = "",col = ifelse(c(1:p)[-nl]%in%s,"blue","grey80"),border=NA,cex.names = 0.6,ylim=c(0,0.6),cex.lab=1.25,cex.axis = 1.25)
lines(x = -0.5:((p-2)*1.5),y = rep(1/(p-length(nl)),(p-2)+1),col = "red",lty=2,lwd=2)
legend("topleft",legend=c(as.expression(bquote(DELTA~"="~.(round(DELTA,3)))),as.expression(bquote("ESS ="~.(round(ESS,2))*"%"))),bty = "n",pch = 20,col = "red")

######################################################################################
######################################################################################
######################################################################################

### Find Fourth Order Centrality by Nullifying the most Associated Predictor Variable ###

### Run the RATE Function ###
top = substring(names(res3$KLD)[order(res3$KLD,decreasing=TRUE)[1]],first = 4)
nl = c(nl,as.numeric(top))
res4 = RATE(X=X,f.draws=fhat.rep,nullify = nl,snp.nms = colnames(X),cores = cores)

### Get the Results ###
rates = res4$RATE
DELTA = res4$Delta
ESS = res4$ESS

### Plot the results with the uniformity line ###
barplot(rates,xlab = "Covariates",ylab=bquote(RATE(tilde(beta)[j]~"|"~tilde(beta)[.(as.integer(nl))]=="0")),names.arg = "",col = ifelse(c(1:p)[-nl]%in%s,"blue","grey80"),border=NA,cex.names = 0.6,ylim=c(0,0.6),cex.lab=1.25,cex.axis = 1.25)
lines(x = -0.5:((p-3)*1.5),y = rep(1/(p-length(nl)),(p-3)+1),col = "red",lty=2,lwd=2)
legend("topleft",legend=c(as.expression(bquote(DELTA~"="~.(round(DELTA,3)))),as.expression(bquote("ESS ="~.(round(ESS,2))*"%"))),bty = "n",pch = 20,col = "red")
