data {
  int<lower=1> N;            // num observations
  int<lower = 0> K;          // num exogenous predictors
  real y[N];                 // observed outputs
  matrix[N, K] exog;         // exogenous series
  int<lower = 0> p;          // ar component
  int<lower = 0> q;          // ma component
  int<lower = 0> max_pq;     // max of p and q
  int<lower = 0> scale;      // standard deviation of priors
}
parameters {
  matrix<lower = -1, upper = 1>[p, K] phi_x;        // exogenous autoregression coeff
  real<lower = -1, upper = 1> phi[p];
  real<lower = -1, upper = 1> theta[q];             // moving avg coeff
  real<lower=0> sigma;       // noise scale
}
model {
  vector[N] nu;              // prediction for time t   
  vector[N] err;             // error for time t
  for(i in 1:p){
    nu[i] = phi[i] + sum(phi_x[i]);
  }
  for(i in 1:q){
    err[i] = 0;
  }
  for (t in 1+max_pq:N){
    for(i in 1:p){
      nu[t] = phi[i]*y[t-i] + dot_product(phi_x[i], exog[t-i]);
    }
    for(i in 1:q){
      nu[t] += theta[i]*err[t-i];
      err[t] = y[t] - nu[t];
    }
  }
  to_vector(phi_x) ~ normal(0, scale);
  phi ~ normal(0, scale);
  theta ~ normal(0, scale);
  sigma ~ cauchy(0, scale);
  err ~ normal(0, sigma);    // likelihood
}