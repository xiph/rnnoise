#!/bin/sh

cargo build \
    && make \
    && ./examples/rnnoise_demo tests/testing.raw out.raw \
    && cargo run --bin corr out.raw tests/reference_output.raw \
    && echo "Check passed!"
