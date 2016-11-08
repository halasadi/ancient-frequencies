data {
  // see page 149 in the manual for reference on NA values
  int <lower=1> O  // number of individuals x number of obs
  int <lower=1> N; // number of individuals
  int <lower=1> P; // number of SNPs
  vector[N] t;
  vector[N] l[2]; // lat, longitude
  int<lower=1, upper=N> jj[O];
  int<lower=1, upper=p> kk[O];
  int<lower=0, upper=1> y[O];
  vector[N] ones;
}


// we have a mean for every SNP
parameters {
  real mu[P];    // mean of vectors for every SNP
  vector[P] theta[N]; // frequencies on the uncontrained scale
}


transformed_parameters {
   real<lower=0> alpha_one;
   real<lower=0> alpha_two;
   real<lower=0> phi;
   real<lower=0> beta;
}

model {   
   matrix[N,N] Sigma;
   for (i in 1:N){
       for (j in i:N) {
       	   Sigma[i,j] = (1/phi) * exp(-1 * distance(l[i],l[j])/alpha_one) * exp(-1 * distance(t[i],t[j])/alpha_2)
	   Sigma[j,i] = Sigma[i,j];
	}
   }
   
   vector[N] muvec;
   for (i in 1:P){
       muvec = ones * mu[i];
       theta[i] ~ multi_normal(muvec, Sigma);
   }
   
   for (i in 1:O){
       y[i] = bernouli_logit(theta[kk[i], jj[i]]);
   }

   // use an heirarchal prior to share information across SNPs
   for (i in 1:P){
       mu[i] ~ normal(0, 1/beta);
   }

   // setting the priors
   beta ~ Gamma(0.001, 0.001);
   phi ~ Gamma(0.001, 0.001);

}
     