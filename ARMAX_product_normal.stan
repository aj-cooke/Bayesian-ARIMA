data {
  int<lower=1> N;            // num observations
  int<lower = 0> K;          // num exogenous predictors
  real y[N];                 // observed outputs
  matrix[N, K] exog;         // exogenous series
  int<lower = 0> p;          // ar component
  int<lower = 0> q;          // ma component
  int<lower = 0> max_pq;     // max of p and q
  int<lower = 0> nn_phi;
  int<lower = 0> nn_exog;
  int<lower = 0> nn_theta;
  int<lower = 0> scale_phi;
  int<lower = 0> scale_exog;
  int<lower = 0> scale_theta;
}
parameters {
  matrix<lower = -1, upper = 1>[p, nn_phi] phis;
  real<lower = -1, upper = 1> phi_xs[p, K, nn_exog];        // exogenous autoregression coeff
  matrix<lower = -1, upper = 1>[q, nn_theta] thetas;             // moving avg coeff
  real<lower=0> sigma;       // noise scale
}
transformed parameters{
  real phi[p];
  real theta[q];
  matrix[p, K] phi_x;
  for(i in 1:p){
    phi[i] = prod(phis[i]);
    for (l in 1:K) phi_x[i,l] = prod(phi_xs[i,l]);
  }
  for(i in 1:q){
    theta[i] = prod(thetas[i]);
  }
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
  for (t in 1+max_pq:N) {
    for(i in 1:p) {
      nu[t] = phi[i]*y[t-i] + dot_product(phi_x[i], exog[t-i]);
    }
    for(i in 1:q){
      nu[t] += theta[i]*err[t-i];
      err[t] = y[t] - nu[t]; 
    }
      
  }          
  for(i in 1:p){
    for(l in 1:K) to_vector(phi_xs[i,l]) ~ normal(0, 5);
  }
  to_vector(phis) ~ normal(0, 5);
  to_vector(thetas) ~ normal(0, 5);
  sigma ~ cauchy(0, 5);
  err ~ normal(0, sigma);    // likelihood
}