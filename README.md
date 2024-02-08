# Sampling-Based Techniques for Training Deep Neural Networks with Limited Computational Resources


`Approx_LSH`: Contains implementation of ALSH-approx. Run `main.py` to see the results. The configuration can be changed in `main.py`.

`Regular_NN`: Contains implementation of vanilla neural network. Run `main.py` to see the results.

`DropOut`: Contains implementations of Dropout and Adaptive Dropout. Run `main.py` to see the results.


Activation function: ReLU; weight initialization: Kaiming initialization

Loss Function: Negative Log-Likelihood


Datasets: MNIST, NORB, Fashion-MNIST

Input and output dimension must change according to the dataset.

| Dataset | Input |   Output |
| ------- | ----- | -------- |
| NORB    | 9216  |    5     |
| MNIST   | 784   |    10    |
| Fashion-MNIST  | 784   |    10    |


Check out MC-approx here:
https://github.com/acsl-technion/approx


Find the supplementary material in `Appendix.pdf`.
