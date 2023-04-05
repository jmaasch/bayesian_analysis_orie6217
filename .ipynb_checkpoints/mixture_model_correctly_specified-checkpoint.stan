data {
    int<lower=0> N;
    vector[N] y;
}

parameters {
    vector[2] mu;
    real<lower=0> sigma;
    real<lower=0,upper=1> theta;
}

/*
Model data as Gaussian mixture where components have equal variance.
*/
model {
    sigma ~ normal(2, 2);
    mu[1] ~ normal(2, 2);
    mu[2] ~ normal(4, 2);
    theta ~ beta(1, 1);
    for (n in 1:N)
        target += log_mix(theta,
                          normal_lpdf(y[n] | mu[1], sigma),
                          normal_lpdf(y[n] | mu[2], sigma));
}

/*
Draw samples from the posterior predictive distribution to check fit.
*/
generated quantities {

    array[N] real y_posterior_pred;
    array[N] int component_id_pred;
    
    for(n in 1:N){
    
        // Predict component membership.
        component_id_pred[n] = bernoulli_rng(theta);
        
        // Sample from posterior.
        y_posterior_pred[n] = normal_rng(mu[component_id_pred[n] + 1], sigma);
    }
    
}