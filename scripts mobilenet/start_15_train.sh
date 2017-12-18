#!/bin/sh

IMAGE_SIZE=224
ARCHITECTURE="mobilenet_1.0_${IMAGE_SIZE}"

for i in `seq 1 15`;
do
	python -m scripts.retrain \
	  --bottleneck_dir=tf_files/bottlenecks \
	  --model_dir=tf_files/models/ \
	  --summaries_dir=tf_files/results/"train_$i"/training_summaries/"${ARCHITECTURE}" \
	  --output_graph=tf_files/results/"train_$i"/retrained_graph.pb \
	  --output_labels=tf_files/results/"train_$i"/retrained_labels.txt \
	  --architecture="${ARCHITECTURE}" \
	  --image_dir=tf_files/images \
	  --testing_percentage=10 \
	  --validation_percentage=10 \
	  --learning_rate=0.01 \
	  --eval_step_interval=100 \
	  --how_many_training_steps=100
done

