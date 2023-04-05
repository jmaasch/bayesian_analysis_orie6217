data {
    int<lower=0> N;
    vector[N] y;
}

/*
Mixture model parameters when components are t-distributions:
- mu: location.
- nu: degrees of freedom.
- sigma: scale.
- theta: probability that component ID = 1.
*/
parameters {
    vector[2] mu;
    real<lower=1,upper=100> nu;
    real<lower=0> sigma;
    real<lower=0,upper=1> theta;
}

/*
Model components as t-distributions with equal variance and degrees of freedom.

The prior distribution for degrees of freedom was selected based on:
https://en.wikipedia.org/wiki/Student%27s_t-distribution#Bayesian_inference:_prior_distribution_for_the_degrees_of_the_freedom
*/
model {
    sigma ~ normal(2, 2);
    mu[1] ~ normal(2, 2);
    mu[2] ~ normal(4, 2);
    theta ~ beta(1, 1);
    nu ~ lognormal(1, 1);
    for (n in 1:N)
        target += log_mix(theta,
                          student_t_lpdf(y[n] | nu, mu[1], sigma),
                          student_t_lpdf(y[n] | nu, mu[2], sigma));
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
        y_posterior_pred[n] = student_t_rng(nu, mu[component_id_pred[n] + 1], sigma);
    }
    
}