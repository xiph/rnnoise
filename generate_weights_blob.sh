#!/bin/sh
echo ##########
echo "Compiling Generator..."
echo ##########

gcc -o generate_weights_blob \
	src/write_weights.c src/parse_lpcnet_weights.c

echo ##########
echo "Generating weights_blob.bin..."
echo ##########

./generate_weights_blob

echo ##########
echo "Generated weights_blob.bin !!"
ls -alh | grep weights_blob.bin
echo ##########
