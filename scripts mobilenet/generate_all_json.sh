#!/bin/sh

LOCAL_TARGET="tf_files/resultados/desbalanceada/"

for i in $LOCAL_TARGET*; do
	echo $(basename $i)
	python -m scripts.evaluate $i/retrained_graph.pb $i.json
done