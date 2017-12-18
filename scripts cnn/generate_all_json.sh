#!/bin/sh

LOCAL_TARGET="results/modelo1/"

for i in $LOCAL_TARGET*.hdf5; do
echo $i;
	python generate_json_result.py $LOCAL_TARGET$(basename $i) output$(basename $i).json
done