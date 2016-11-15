// Fit a Gaussian process's hyperparameters
// for squared exponential prior

data {
  int<lower=1> N;
  int<lower=1> P; // number of replicates
  vector[N] x;
  vector[N] y[P];
}
transformed data {
  vector[N] mu;
  for (i in 1:N) 
    mu[i] <- 0;
}
parameters {
  real<lower=0> eta_sq;
  real<lower=0> rho_sq;
  real<lower=0> sigma_sq;
}
model {
  matrix[N,N] Sigma;

  // off-diagonal elements
  for (i in 1:(N-1)) {
    for (j in (i+1):N) {
      Sigma[i,j] <- eta_sq * exp(-rho_sq * pow(x[i] - x[j],2));
      Sigma[j,i] <- Sigma[i,j];
    }
  }

  // diagonal elements
  for (k in 1:N)
    Sigma[k,k] <- eta_sq + sigma_sq; // + jitter
    
  for (n in 1:P)
    y[n] ~ multi_normal(mu, Sigma);

  #eta_sq ~ cauchy(0,5);
  #rho_sq ~ cauchy(0,5);
  #sigma_sq ~ cauchy(0,5);
  eta_sq ~ gamma(1,1);
  rho_sq ~ gamma(1,1);
  sigma_sq ~ gamma(1,1);

}
