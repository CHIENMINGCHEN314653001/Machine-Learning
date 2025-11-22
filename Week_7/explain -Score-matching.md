**Explain the abstraction of score matching**

**Ⅰ. Abstraction of Score Matching**

In generative modeling, the goal is to learn the data probability density function $$p(x)$$.  
However, directly estimating $$p(x)$$ is difficult because it involves the normalization constant

$$Z(\theta) = \int e^{q(x;\theta)} dx$$

,the integral above is very difficult to calculate (because of dimensionality).

To overcome this, score matching  avoids computing $$p(x)$$ by instead learning the score function:

$$S(x) = \nabla_x \log p(x)$$

,which is the gradient of the log-density with respect to $$x$$.  

The score function points toward regions of higher data density.

<br></br>
**Score matching is divided into three types:**
  * Explicit score matching (ESM)
  * Implicit score matching (ISM)
  * Denoising Score Matching (DSM)

<br></br>
**Ⅱ. Explicit Score Matching(ESM) and Implicit Score Matching(ISM)**

The ESM objective minimizes the difference between the true and estimated score functions:


$$L_{\text{ESM}}(\theta) = \mathbb{E}_{x\sim p(x)} \| S(x;\theta) - \nabla_x \log p(x) \|^2.$$

Since $$\nabla_x \log p(x)$$ is unknown, we can't compute this directly( $$\because \quad p(x)$$ is difficult to estimate).

Using integration by parts, this can be rewritten as the ISM loss:


$$L_{\text{ISM}}(\theta) = \mathbb{E}_{x\sim p(x)} \left[\|S(x;\theta)\|^2 + 2\nabla_x \cdot S(x;\theta)\right]$$

,which does not require knowledge of $$\log p(x)$$.  
Thus, $$L_{\text{ISM}}$$ and $$L_{\text{ESM}}$$ are equivalent in optimization.
However, the original objective function (such as Implicit Score Matching,ISM) contains $$\text{Tr}(\nabla_x \mathbf{s}_\theta(x))$$ , which involves expensive second-order derivative computations in high-dimensional data, making it difficult to apply in practice.


<br></br>

**Ⅲ. Denoising Score Matching (DSM)**

In practical applications, the data distribution $$𝑝(𝑥)$$ may be discrete or lie on a low-dimensional manifold, which makes it difficult to estimate $$𝑆(𝑥)$$.
To solve this, we add Gaussian noise to data:


$$x = x_0 + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

,and define the noisy distribution $$p_\sigma(x)$$.  
Then, the denoising score matching loss becomes:

$$L_{\text{DSM}}(\theta) =  \mathbb{E}_{x_0\sim p_0(x_0)}\mathbb{E}_{x|x_0\sim p_\sigma(x|x_0)}\frac{1}{\sigma^2}\left\|\sigma S_\sigma(x_0+\sigma\epsilon;\theta)+\epsilon \right\|^2.$$


This allows us to train a neural network  $$S_\sigma(x;\theta)$$  to approximate  $$\nabla_x \log p_\sigma(x)$$.

<br></br>

**Ⅳ.Score-Based Generative Models**

In the forward process, a Markov chain or stochastic differential equation (SDE) is defined to gradually add noise to real data $x_0$ at different scales until it becomes pure Gaussian noise $x_T$.  
This process systematically diffuses the data distribution into a noise distribution across multiple noise levels:

$$\sigma_1 > \sigma_2 > \cdots > \sigma_N$$

During training, the model learns to approximate the score function of the noisy data distribution $p_t(x)$ at each time step $t$:

$$\mathbf{s}_\theta(x_t, t)$$

The training objective is essentially a weighted sum of DSM losses, allowing the model to learn the denoising direction at each noise level.  
In other words, the model learns a family of score functions:

$$S_{\sigma_t}(x)$$

that capture the gradient information of the data distribution at various noise scales.

In the reverse generation process, we start from pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$ and iteratively denoise it using the learned score function $\mathbf{s}_\theta(x_t, t)$.  
This process can be implemented using Langevin Dynamics or by solving the reverse SDE/ODE, typically expressed as:

$$x_{t-1} = x_t + \epsilon \, S_{\sigma_t}(x_t) + \sqrt{2\epsilon}\, z, \quad z \sim \mathcal{N}(0, I)$$

This is equivalent to Langevin dynamics guided by the learned score function.  Through iterative refinement, random noise is gradually transformed into realistic data samples.

<br></br>
 
**Conclusion**
* Score Matching provides a theoretical framework for learning data distributions from gradients rather than densities themselves.
* Denoising Score Matching introduces controllable noise, transforming the computationally difficult-to-calculate gradient estimation into a noise prediction task that is easy to train, fundamentally addressing its practicality.
* The Diffusion Model generalizes the concept of the DSM to multiple timescales (multiple noise levels) and utilizes the learned score function (i.e., the denoising direction) to guide an efficient and stable inverse sampling process, achieving state-of-the-art generation capabilities.
