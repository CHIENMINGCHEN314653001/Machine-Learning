**Report on Lemma 3.1 and Lemma 3.2** 
*Based on Ryck et al., “On the approximation of functions by tanh neural networks”*

<br></br>

**Ⅰ. Background**

This part studies how tanh neural networks (networks using the hyperbolic tangent activation function) can approximate mathematical functions. A key step is showing that neural networks can approximate polynomials and basic power functions like $$x^2, x^3, \dots\ $$.

- Many functions can be built from polynomials (think of Taylor expansions).
- If a neural network can approximate powers of a variable, it can be combined to approximate much more complex functions.

<br></br>

 **Ⅱ. Lemma 3.1 — Approximating Odd Powers**

**Formal Statement (simplified):**
For any integer s and tolerance  $$\varepsilon > 0\ $$, there exists a **shallow tanh neural network** (just one hidden layer) that can approximate the functions

$$
\ f_p(x) = x^p, \quad p = 1, 3, 5, \dots, s\
$$

(i.e., all *odd powers* up to $$s$$) on a bounded interval $$[-M, M]$$. The error of approximation can be made smaller than $$\varepsilon$$.

**Idea:**
- The proof uses the fact that the derivative of the tanh function behaves like a polynomial around certain points.
- By carefully shifting and scaling tanh, the network can mimic  $$x, x^3, x^5, \dots $$.
- The construction uses only a small number of neurons in one layer, so the network remains simple (“shallow”).

**Intuitive Explanation:**
Imagine you want to draw the curve of  $$y = x^3\ $$. Instead of using the exact formula, you use combinations of squished S-shaped curves $$tanh$$. If you add them together in the right way, you can make a curve that looks very close to $$x^3\$$. Lemma 3.1 guarantees that this is always possible, no matter how high the odd power is.

<br></br>

**Ⅲ. Lemma 3.2 — Approximating Even Powers**

**Formal Statement (simplified):**
For any odd integer $$s\$$ and tolerance $$\varepsilon > 0$$, there exists a shallow tanh neural network that can approximate the functions

$$
 f_p(x) = x^p, \quad p = 0, 2, 4, \dots, s\
$$  

(i.e., all *even powers* up to $$s\$$, including the constant function $$\(1\)$$) on $$[-M, M]$$. Again, the error can be made smaller than $$\(\varepsilon\)$$.

**Idea:**
- Even powers are trickier because tanh is an odd function (it’s symmetric about the origin).
- To get even powers, the authors use an algebraic trick:

$$
    y^{2n} = \frac{1}{2\alpha(2n+1)}\Big((y+\alpha)^{2n+1} - (y-\alpha)^{2n+1} - \text{(lower order terms)}\Big).
$$

- This formula expresses an even power as a combination of odd powers, which Lemma 3.1 already covers.
- By recursion, the network can build up $$x^2, x^4, \dots\$$.

**Intuitive Explanation:**
Think of $$x^2$$. Since tanh naturally gives odd-like shapes, it’s not easy to directly approximate $$\(x^2\)$$. But if you cleverly combine approximations of odd powers like \(x^3\) and $$\(x^1\)$$, you can recover \(x^2\). Lemma 3.2 shows how to systematically build even powers this way.

<br></br>


**Ⅳ. Summary**
- Together, Lemma 3.1 and 3.2 show that a shallow tanh neural network can approximate any polynomial function (odd or even powers).
- Since polynomials can approximate any continuous function on a bounded interval (Weierstrass Approximation Theorem), these lemmas are the building blocks for proving that tanh networks are powerful universal approximators.
- Lemma 3.1: Shallow tanh networks can approximate odd powers ($$x, x^3, x^5, \dots$$).
- Lemma 3.2: By combining odd powers, they can also approximate even powers ($$1, x^2, x^4, \dots$$).
- Comprehensive: Any polynomial can be approximated by a shallow tanh neural network.
- By extension: Since polynomials approximate all continuous functions (Weierstrass theorem), tanh networks serve as universal approximators.
- Key: Even with only one hidden layer, tanh neural networks still have the mathematical power to approximate any function.

<br></br>

Example link:https://colab.research.google.com/drive/1D9K4-884yM9TjwaHkSyakIfExqYZdmLn?usp=sharing

<br></br>
