data {
  // see page 149 in the manual for reference on NA values
  int <lower=1> N; // number of individuals
  int <lower=1> P; // number of SNPs
  vector[N] t;
  vector[N] l[2]; // lat, longitude
  int<lower=0, upper = 1> y[P,N];
  //vector[N] y[P]; // matrix of counts
}

transformed data {
  vector[N] ones;
  for (i in 1:N) ones[i] = 1;
}
// we have a mean for every SNP
parameters {
  real mu[P];    // mean of vectors for every SNP
  real<lower=0, upper = 1> alpha;
  real<lower=0, upper = 1> beta;
  real<lower=0> phi;
  real<lower=0> sigma_sq; // nugget effect
  vector[N] theta[P]; // frequencies on the uncontrained scale
}

model {   
   matrix[N,N] Sigma;
   for (i in 1:(N-1)){
       for (j in (i+1):N) {
       	   Sigma[i,j] = phi * exp(-sqrt(square(t[i]-t[j]))/beta); // * exp(-distance(l[i],l[j])/alpha);
       	   Sigma[j,i] = Sigma[i,j];
	}
   }
   
   for (k in 1:N){
     Sigma[k,k] = phi + sigma_sq; // + nugget effect
   }
   
   for (n in 1:P){
       theta[n] ~ multi_normal(ones*mu[n], Sigma);
       y[n] ~ bernoulli_logit(theta[n]);
   }
   
   phi ~ gamma(0.01, 0.01);
   alpha ~ gamma(0.01, 0.01);
   beta ~ gamma(0.01, 0.01);
}
     