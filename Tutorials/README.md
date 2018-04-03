# Demonstrations and Tutorials on Running RATE
In the file entitled "Centrality_Tutorial.R", we walk through the implementation of computing/utilizing distributional centrality measures. Specifically, we describe in detail: (1) how to compute a covariance matrix using the Gaussian kernel function; (2) how to fit a standard Bayesian Gaussian process (GP) regression model; and (3) prioritizing variables via their first, second, third, and fourth order distributional centrality. This script is based on a simple (and small) genetics example where we simulate genotype data for n = 250 individuals with p = 500 measured genetic variants. We randomly assume the last three predictors j* = {23, 24, 25} are causal and have true association with the generated (continuous) phenotype y. We then assume that the j* predictor variables explain a fixed H<sup>2</sup>% (phenotypic variance explained; PVE) of the total variance in the response V(y). This parameter H<sup>2</sup> can alternatively be described as a factor controlling the signal-to-noise ratio. The parameter rho represents the proportion of H<sup>2</sup> that is contributed by additive effects versus interaction effects. Namely, the additive effects make up rho%, while the pairwise interactions make up the remaining (1 − rho)%.

In the "Power_Comparisons.R" file, we demonstrate the power of distributional centrality via RATE measures. Here, we focus on simulated genotype data for n = 500 individuals with p = 2500 measured genetic variants. Next, we compare our approach to: (1) L1- regularized lasso regression; (2) the combined regularization utilized by the elastic net; (3) a genome scan with individual single nucleotide polymorphisms (SNPs) fit via a univariate linear model (SCANONE); and (4) a commonly used spike and slab prior model, also known as Bayesian variable selection regression, which computes posterior inclusion probabilities (PIPs) for each covariate as a mixture of a point mass at zero and a diffuse normal centered around zero. This script is based on the simulations from Crawford et al. (2018). We assess the association mapping ability of each method with (i) different signal-to-noise ratios in H<sup>2</sup>, (ii) varying levels of additive and interaction effects in rho, and (iii) with data affected by population stratification.

### R Packages Required for RATE Tutorials
The RATE tutorial and examples require the additional installation of the following R libraries:

[glmnet](https://cran.r-project.org/web/packages/glmnet/index.html)

[varbvs](https://cran.r-project.org/web/packages/varbvs/index.html)

Once again, the easiest method to install these packages is with the following example command entered in an R shell:

install.packages("glmnet", dependecies = TRUE)

Alternatively, one can also [install R packages from the command line](http://cran.r-project.org/doc/manuals/r-release/R-admin.html#Installing-packages).
