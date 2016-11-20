// Fit a Gaussian process's hyperparameters
// for squared exponential prior

data {
  int<lower=1> N; // number of individuals
  vector[N] t; // time
  matrix[N,2] x; // lat longs
  int<lower=0> n[N]; // samples per location
  int<lower=0> P; // number of SNPs
  int<lower=0, upper = 30> y[P, N]; // assumes max number of haploids is 30
}
transformed data {
  vector[N] mu;
  for (i in 1:N) 
    mu[i] <- 0;
}
parameters {
  real<lower=0> phi;
  matrix[P,N] theta;
  real<lower=0> alpha;
  real<lower=0> beta;
}

model {
  matrix[N,N] Sigma;

  // off-diagonal elements
  for (i in 1:(N-1)) {
    for (j in (i+1):N) {
      real d;
      d = distance(x[i], x[j]);
      Sigma[i,j] <- phi * exp(-d/alpha) * exp(-sqrt(pow(t[i] - t[j],2))/beta);
      Sigma[j,i] <- Sigma[i,j];
    }
  }

  // diagonal elements
  for (k in 1:N)
    Sigma[k,k] <- phi + 0.01; // + jitter
  
  for (k in 1:P){
    theta[k] ~ multi_normal(mu, Sigma);
    y[k] ~ binomial_logit(n, theta[k]);
  }

  alpha ~ gamma(5,5);
  beta ~ gamma(5,5);
  phi ~ gamma(1,1);
}
