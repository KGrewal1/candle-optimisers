# Changelog

## v0.3.1 (2023-12-20)

* Improved Documentation
* Add ability to set more optimiser parameters (see issue <https://github.com/huggingface/candle/issues/1448> regarding LR schedulers in `candle`)
* All params are now `Clone`, `PartialEq` and `PartialOrd`

## v0.3.0 (2023-12-07)

* Renamed to candle-optimisers for release on crates.io
* Added fuller documentation
* Added decoupled weight decay for SGD

## v0.2.1 (2023-12-06)

* Added decoupled weight decay for all adaptive methods

## v0.2.0 (2023-12-06)

* changed weight decay to `Option` type as opposed to checking for 0
* made `enum` for decoupled weight decay and for momentum
* added weight decay to LBFGS

## v0.1.x

* Initial release and adddition of features
