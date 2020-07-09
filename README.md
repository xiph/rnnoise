# nnnoiseless

`nnnoiseless` is a rust crate for suppressing audio noise. It is a (safe) rust port of
the [`RNNoise`](https://github.com/xiph/rnnoise) C library, and is based on a recurrent
neural network.

While `nnnoiseless` is meant to be used as a library, a simple command-line
tool is provided as an example. It operates on RAW 16-bit little-endian mono
PCM files sampled at 48 kHz. It can be used as:

```
cargo run --release --example rnnoise_demo INPUT.raw OUTPUT.raw
```

The output is also a 16-bit raw PCM file.
