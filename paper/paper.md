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
 - name: Yusuf Hamied Department of Chemistry, University of Cambridge
   index: 1
date: 20 December 2023
bibliography: paper.bib
---

# Summary

`candle-optimisers` is a crate for optimisers written in Rust for use with candle (@candle) a lightweight machine learning framework. The crate offers a set of
optimisers for training neural networks. This allows network training to be done with far lower overhead than using a full python framework such as PyTorch or Tensorflow.

# Statement of need

Rust provides the opportunity for the development of high performance machine learning libraries, with a leaner runtime. However, there is a lack of optimisation algorithms implemented in Rust,
with libraries currently implementing only some combination of Adam, AdamW, SGD and RMSProp.
This crate aims to provide a set of complete set of optimisation algorithms for use with candle.
This will allow Rust to be used for the training of models more easily.

# Features

This library implements the following optimisation algorithms:

* SGD (including momentum and Nesterov momentum (@nmomentum))

* AdaDelta (@adadelta)

* AdaGrad (@adagrad)

* AdaMax (@adam)

* Adam (@adam) including AMSGrad (@amsgrad)

* AdamW (@weightdecay) (as decoupled weight decay of Adam)

* NAdam (@nadam)

* RAdam (@radam)

* RMSProp (@rmsprop)

* LBFGS (@LBFGS)

Furthermore, decoupled weight decay (@weightdecay) is implemented for all of the adaptive methods listed and SGD,
allowing for use of the method beyond solely AdamW.

# References