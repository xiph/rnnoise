#!/bin/sh
set -e

hash=`cat model_version`
model=rnnoise_data-$hash.tar.gz

if [ ! -f $model ]; then
        echo "Downloading latest model"
        wget https://media.xiph.org/rnnoise/models/$model
fi

if command -v sha256sum
then
   echo "Validating checksum"
   checksum="$hash"
   checksum2=$(sha256sum $model | awk '{print $1}')
   if [ "$checksum" != "$checksum2" ]
   then
      echo "Aborting due to mismatching checksums. This could be caused by a corrupted download of $model."
      echo "Consider deleting local copy of $model and running this script again."
      exit 1
   else
      echo "checksums match"
   fi
else
   echo "Could not find sha256 sum; skipping verification. Please verify manually that sha256 hash of ${model} matches ${1}."
fi


tar xzvomf $model

