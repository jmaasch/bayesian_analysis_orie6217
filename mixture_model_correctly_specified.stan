/*
# MIXTURE MODELS
https://betanalpha.github.io/assets/case_studies/mixture_models.html
https://mc-stan.org/users/documentation/case-studies/identifying_mixture_models.html
https://github.com/nikhgarg/2023_ADBA_PhDclass/blob/main/lecture7_mixtureofnormals.ipynb
https://mc-stan.org/docs/stan-users-guide/summing-out-the-responsibility-parameter.html

# VI
https://mc-stan.org/docs/cmdstan-guide/variational-inference.html
https://mc-stan.org/docs/cmdstan-guide/variational-inference-algorithm-advi.html
https://cmdstanpy.readthedocs.io/en/v0.9.77/examples/Variational%20Inference.html
https://cmdstanpy.readthedocs.io/en/stable-0.9.65/variational_bayes.html

# MCMC
https://mc-stan.org/docs/cmdstan-guide/mcmc-config.html
https://mc-stan.org/docs/cmdstan-guide/mcmc-intro.html

# DISTRIBUTIONS
https://mc-stan.org/docs/2_18/functions-reference/student-t-distribution.html
*/
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
https://mc-stan.org/docs/cmdstan-guide/gc-intro.html
*/
generated quantities {

    array[N] real y_posterior_pred;
    array[N] int component_id_pred;
    
    for(n in 1:N){
    
        // Predict component membership.
        component_id_pred[n] = bernoulli_logit_rng(theta);
        
        // Sample from posterior.
        y_posterior_pred[n] = normal_rng(mu[component_id_pred[n] + 1], sigma);
    }
    
}