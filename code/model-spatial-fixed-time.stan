data {
  // see page 149 in the manual for reference on NA values
  int <lower=1> O;  // total number of individuals
  int <lower=1> N; // number of individuals
  int <lower=1> P; // number of SNPs
  matrix[N,2] x; // lat longs
  int<lower=1, upper=N> jj[O]; // observation -> individ
  int<lower=1, upper=P> kk[O]; // observation -> snp
  int<lower=0> y[O];
  int<lower=0> n[N]; // samples per location
  vector[N] t;
  real<lower=0> phi;
  real<lower=0> alpha;
}


transformed data {
  vector[N] ones;
  for (i in 1:N) 
    ones[i] <- 1;
}

// we have a mean for every SNP
parameters {
  vector[P] mu;    // mean for every SNP
  matrix[P,N] theta;
  real<lower=0> eta_sq;
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
   
   for (i in 1:P){
     theta[i] ~ multi_normal(ones * mu[i], Sigma);
   }
   
   for (i in 1:O){
     y[i] ~ binomial_logit(n[jj[i]], theta[kk[i], jj[i]]);
   }

   // use an heirarchal prior to share information across SNPs
   for (i in 1:P){
     mu[i] ~ normal(0, 1/eta_sq);
   }

   // setting the priors
   eta_sq ~ gamma(0.001, 0.001);
   beta ~ gamma(5,5);
}
