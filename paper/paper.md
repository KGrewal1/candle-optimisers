---
title: 'Candle Optimisers: A Rust crate for optimisation algorithms'
tags:
  - Rust
  - optimisation
  - optimization
  - machine learning
authors:
  - name: Kirpal Grewal
    orcid: 0009-0001-7923-9975
    affiliation: 1
affiliations:
 - name: University of Cambridge
   index: 1
date: 12 December 2023
bibliography: paper.bib
---

# Summary

`candle-optimisers` is a crate for optimisers written in Rust for use with [candle] a lightweight machine learning framework. The library offers a set of
optimisers for training neural networks. This allows network training to be done with far lower overhead than using a full python framework such as PyTorch or Tensorflow.

# Statement of need

Rust provides the opportunity for the development of high performance machine learning libraries, with a leaner runtime. However, there is a lack of optimisation algorithms implemented in Rust,
with libraries currently implementing only some combination of Adam, AdamW, SGD and RMSProp.
This crate aims to provide a set of complete set of optimisation algorithms for use with [candle].
This will allow Rust to be used for the training of models more easily as well as the deployment of inference models.

# Features

This library implements the following optimisation algorithms:

* SGD

* RMSprop

* AdaDelta

* AdaGrad

* AdaMax

* Adam

* AdamW (as decoupled weight decay of Adam)

* NAdam

* RAdam

* RMSProp

* LBFGS

Furthermore, decoupled weight decay [Decoupled Weight Decay Regularization](https://arxiv.org/pdf/1711.05101.pdf) is implemented for all of the adaptive methods listed and SGD,
allowing for use of the method beyond solely AdamW.

# References