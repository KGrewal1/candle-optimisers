# Optimisers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A crate for optimisers for use with [candle](https://github.com/huggingface/candle), the minimalist ML framework

* Momentum enhanced SGD

* AdaGrad

* AdaDelta

* AdaMax

* NAdam

* RAdam

* RMSprop

These are all checked against their pytorch implementation (see pytorch_test.ipynb) and should implement the same functionality (though without some input checking).

Currently unimplemented from pytorch:

* AdamW (see Adam in candle-nn)

* SparseAdam (unsure how to treat sparse tensors in candle)

* ASGD (no pseudocode)

* LBFGS (need to reformulate in terms of tensors / no pseudocode)

* Rprop (need to reformulate in terms of tensors)
