// Fit a Gaussian process's hyperparameters
// for squared exponential prior

data {
  int<lower=1> N;
  vector[N] x;
  matrix[N,2] t;
  int<lower=0> n[N];
  int<lower=0> P;
  int<lower=0, upper = 10> y[P, N];
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
      d = distance(t[i], t[j]);
      Sigma[i,j] <- phi * exp(-sqrt(pow(x[i] - x[j],2))/alpha)*exp(-d/beta);
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

  alpha ~ gamma(0.1,0.1);
  beta ~ gamma(0.1,0.1);
  phi ~ gamma(0.1,0.1);
}
