Consider a sequence of functions $f_n(x)$ defined recursively as:

$$
f_1(x) = \tanh(x)
$$

$$
f_n(x) = \tanh \left( f_{n-1}(x) \right) \quad \text{for } n \ge 2
$$

find

$$
L(x) = \lim_{n \to \infty} f_n(x)=？
$$

If the input $x$ represents the weighted sum in an infinitely deep neural network where every layer uses the $\tanh$ activation, what is the significance of the limiting function $L(x)$ in terms of the network's output?
