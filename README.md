# Optimisers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A crate for optimisers for use with [candle](https://github.com/huggingface/candle), the minimalist ML framework

* Momentum enhanced SGD

* AdaGrad

* AdaDelta

* AdaMax

* NAdam

These are all checked against their pytorch implementation and should implement the same functionality (though without some input checking).
