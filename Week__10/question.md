In the context of ESM for diffusion models, the weighting term $$\lambda(t)$$ is often chosen proportional to the noise scale $$\sigma_t$$  to balance contributions from different noise levels.  

Is it possible to determine a more optimal scaling factor $$c$$ such that  

$$\lambda(t) = c\sigma_t$$

(or another functional multiple of $$\sigma_t$$ ) minimizes the overall training variance or improves convergence stability?

Then, can we find an optimal constant or functional multiplier $$c(t)$$ such that  

$$
\lambda(t) = c(t)\sigma_t
$$

leads to improved gradient efficiency or model performance in score-based diffusion training?






