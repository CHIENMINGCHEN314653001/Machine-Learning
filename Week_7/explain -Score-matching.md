 ### Explain the abstraction of score matching

**Ⅰ. Abstraction of Score Matching**

In generative modeling, the goal is to learn the data probability density function $$p(x)$$.  
However, directly estimating $$p(x)$$ is difficult because it involves the normalization constant

$$Z(\theta) = \int e^{q(x;\theta)} dx$$

,the integral above is very difficult to calculate (because of dimensionality).

To overcome this, score matching  avoids computing $$p(x)$$ by instead learning the score function:

$$S(x) = \nabla_x \log p(x)$$

,which is the gradient of the log-density with respect to $$x$$.  

The score function points toward regions of higher data density.

---
**Score matching is divided into three types:**
  * Explicit score matching (ESM)
  * Implicit score matching (ISM)
  * Denoising Score Matching (DSM)
---
**Ⅱ. Explicit and Implicit Score Matching**

The ESM objective minimizes the difference between the true and estimated score functions:


$$L_{\text{ESM}}(\theta) = \mathbb{E}_{x\sim p(x)} \| S(x;\theta) - \nabla_x \log p(x) \|^2.$$

Since $$\nabla_x \log p(x)$$ is unknown, we can't compute this directly( $$\because \quad p(x)$$ is difficult to estimate).

Using integration by parts, this can be rewritten as the ISM loss:


$$L_{\text{ISM}}(\theta) = \mathbb{E}_{x\sim p(x)} \left[\|S(x;\theta)\|^2 + 2\nabla_x \cdot S(x;\theta)\right]$$

,which does not require knowledge of $$\log p(x)$$.  
Thus, $$L_{\text{ISM}}$$ and $$L_{\text{ESM}}$$ are equivalent in optimization.
However, the original objective function (such as Implicit Score Matching,ISM) contains $$\text{Tr}(\nabla_x \mathbf{s}_\theta(x))$$ , which involves expensive second-order derivative computations in high-dimensional data, making it difficult to apply in practice.


---

**Ⅲ. Denoising Score Matching (DSM)**

In practical applications, the data distribution $$𝑝(𝑥)$$ may be discrete or lie on a low-dimensional manifold, which makes it difficult to estimate $$𝑆(𝑥)$$.
To solve this, we add Gaussian noise to data:


$$x = x_0 + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

,and define the noisy distribution $$p_\sigma(x)$$.  
Then, the denoising score matching loss becomes:

$$L_{\text{DSM}}(\theta) =  \mathbb{E}_{x_0\sim p_0(x_0)}\mathbb{E}_{x|x_0\sim p_\sigma(x|x_0)}\frac{1}{\sigma^2}\left\|\sigma S_\sigma(x_0+\sigma\epsilon;\theta)+\epsilon \right\|^2.$$


This allows us to train a neural network  $$S_\sigma(x;\theta)$$  to approximate  $$\nabla_x \log p_\sigma(x)$$.

---

**Ⅳ.Score-Based Generative Models**

In score-based diffusion models, noise is gradually added to data through a forward diffusion process, producing multiple noise levels 

$$\sigma_1 > \sigma_2 > \dots > \sigma_N.$$

The model learns a family of score functions $$S_\sigma(x)$$ for different noise levels using DSM.

During generation (reverse process), we start from pure Gaussian noise and iteratively denoise samples using the learned scores:

$$x_{t-1} = x_t + \epsilon S_{\sigma_t}(x_t) + \sqrt{2\epsilon}\, z, \quad z \sim \mathcal{N}(0, I)$$

,which is equivalent to Langevin dynamics guided by the learned score function.  
This process gradually transforms random noise into realistic data samples.

---
 
**Conclusion**
* Score Matching provides a theoretical framework for learning data distributions from gradients rather than densities themselves.
* Denoising Score Matching introduces controllable noise, transforming the computationally difficult-to-calculate gradient estimation into a noise prediction task that is easy to train, fundamentally addressing its practicality.
* The Diffusion Model generalizes the concept of the DSM to multiple timescales (multiple noise levels) and utilizes the learned score function (i.e., the denoising direction) to guide an efficient and stable inverse sampling process, achieving state-of-the-art generation capabilities.
