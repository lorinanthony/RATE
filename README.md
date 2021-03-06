# Variable Prioritization for Black Box Methods via RelATive cEntrality (RATE)
Our ability to build good predictive models has, in many cases, outstripped our ability to extract interpretable information about the relevance of the input covariates being used. The central aim of [Crawford et al. (2019)](https://arxiv.org/abs/1801.07318) and [Ish-Horowicz et al. (2019)](https://arxiv.org/abs/1901.09839) is to assess variable importance after having fit a nonlinear or nonparametric (Bayesian) model. In this work, we propose a new "RelATive cEntrality" (RATE) measure as an interpretable way to summarize the importance of covariates. By assessing entropy in the joint posterior distribution via Kullback-Leibler divergence (KLD), we can correctly prioritize candidate variables which are not just marginally important, but also those whose associations stem from a significant covarying relationship with other variables in the data. We demonstrate our proposed approach in the context of statistical genetics, where the discovery of variants that are involved in nonlinear interactions is of particular interest. In the `Tutorials` directory, we focus on illustrating RATE through Gaussian process (GP) regression; although, methodological innovations can easily be applied to other machine learning-type methods such as (deep) neural networks as demonstrated in the `Deep Learning` directory. It is well known that nonlinear methods often exhibit greater predictive accuracy than linear models, particularly for outcomes generated by complex data architectures. With simulations and real data examples, we show that applying RATE enables an explanation for this improved performance.

RATE is implemented as a set of parallelizable routines, which can be carried out within an R environment.  Detailed derivations of the algorithm, which utilizes low-rank matrix factorizations for a more practical implementation, are derived in [Supplementary Material](http://lcrawlab.com/Papers/RATE_SI.pdf) of Crawford et al. (2019).

### R Packages Required for RATE
The RATE function software requires the installation of the following R libraries:

* [BAKR](https://github.com/lorinanthony/BAKR) (via GitHub)

* [corpcor](https://cran.r-project.org/web/packages/corpcor/index.html)

* [doParallel](https://cran.r-project.org/web/packages/doParallel/index.html)

* [MASS](https://cran.r-project.org/web/packages/MASS/index.html)

* [Matrix](https://cran.r-project.org/web/packages/Matrix/index.html)

* [Rcpp](https://cran.r-project.org/web/packages/Rcpp/index.html)

* [RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo/index.html)

* [RcppParallel](https://cran.r-project.org/web/packages/RcppParallel/index.html)

Unless stated otherwise, the easiest method to install many of these packages is with the following example command entered in an R shell:

    install.packages("corpcor", dependecies = TRUE)

Alternatively, one can also [install R packages from the command line](http://cran.r-project.org/doc/manuals/r-release/R-admin.html#Installing-packages).

### C++ Functions Required for GP Regression
The code in this repository assumes that basic C++ functions and applications are already set up on the running personal computer or cluster. If not, the functions and necessary Rcpp packages to build nonlinear covariance matrices (e.g. BAKR) and fit a GP regression model will not work properly. A simple option is to use [gcc](https://gcc.gnu.org/). macOS users may use this collection by installing the [Homebrew package manager](http://brew.sh/index.html) and then typing the following into the terminal:

    brew install gcc

For macOS users, the Xcode Command Line Tools include a GCC compiler. Instructions on how to install Xcode may be found [here](http://railsapps.github.io/xcode-command-line-tools.html). For extra tips on how to run C++ on macOS, please visit [here](http://seananderson.ca/2013/11/18/rcpp-mavericks.html). For tips on how to avoid errors dealing with "-lgfortran" or "-lquadmath", please visit [here](http://thecoatlessprofessor.com/programming/rcpp-rcpparmadillo-and-os-x-mavericks-lgfortran-and-lquadmath-error/).

### Demonstrations and Tutorials for Running RATE

In the `Tutorials` directory, we provide a few example scripts that demonstrate how to conduct variable selection in nonlinear models with RATE measures. Here, we consider a simple (and small) genetics example where we simulate genotype data for _n_ individuals with _p_ measured genetic variants. We then randomly select a small number of these predictor variables to be causal and have true association with the generated (continuous) phenotype. These scripts are meant to illustrate proof of concepts and specifically walk through: (1) how to compute a covariance matrix using the Gaussian kernel function; (2) how to fit a standard Bayesian Gaussian process (GP) regression model; and (3) prioritizing variables via their first, second, third, and fourth order distributional centrality.

In the `Deep Learning` directory, we demonstrate how to implement RATE with Bayesian neural network architectures. Notebooks are provided to give explicit details on training procedures and how to determine variable importance for the input features of networks.

### Relevant Citations
L. Crawford, S.R. Flaxman, D.E. Runcie, and M. West (2019). Variable prioritization in nonlinear black box methods: a genetic association case study. _Annals of Applied Statistics_. **13**(2): 958-989.

J. Ish-Horowicz*, D. Udwin*, S.R. Flaxman, S.L. Filippi, and L. Crawford (2019). Interpreting deep neural networks through variable importance. _arXiv_. 1901.09839.

### Questions and Feedback
For questions or concerns with the RATE functions, please contact [Lorin Crawford](mailto:lorin_crawford@brown.edu).

We appreciate any feedback you may have with our repository and instructions.
