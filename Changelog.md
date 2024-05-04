# Changelog

## v0.5.0 (2024-02-28)

* Bump candle requirtement to 0.5.0: this is considered a breaking change due to the reliance of this library on candle-core and candle-nn
* Internal changes for LBFGS line search

## v0.4.0 (2024-02-28)

* Bump candle requirtement to 0.4.0: this is considered a breaking change due to the reliance of this library on candle-core and candle-nn
* Explicit reliance on the candle crates hosted on crates.io : as cargo does not support git dependecies in published crates, this library now points only to the crates.io releases (previously cargo would default to the crates.io instead of git repo anyway: if the git repo is specifically desired this can be obtained by patching the `Cargo.toml` to point at the candle repo)
* Remove intel-mkl feature: features in this library are mainly used for running examples: any code that uses this library should instead use the features directly from the candle crates

## v0.3.2 (2024-01-07)

* move directional evaluation into stronge wolfe
* fix strong wolfe condition when used with weight decay

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
