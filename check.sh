#!/bin/sh

make && ./examples/rnnoise_demo tests/testing.raw out.raw && diff out.raw tests/reference_output.raw && echo "Check passed!"
