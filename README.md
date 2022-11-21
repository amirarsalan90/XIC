# Robust Learning with the Hilbert-Schmidt Independence Criterion

This repository contains a pytorch implementation of HSIC-loss used in the paper "Through a fair looking-glass: mitigating bias in image datasets": https://arxiv.org/abs/2209.08648

If ```x,y``` represent two batches of samples from random variables ```X,Y```, calling
```
HSIC(x,y)
```
would compute the Hilbert-Schmidt Independence Criterion between them.
This code uses Gaussian kernels.
