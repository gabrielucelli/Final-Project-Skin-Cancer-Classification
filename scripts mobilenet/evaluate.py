#!/usr/bin/python
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse

import numpy as np
import PIL.Image as Image
import tensorflow as tf
import itertools
import json

import scripts.retrain as retrain
from scripts.count_ops import load_graph

def evaluate_graph(graph_file_name):

    with load_graph(graph_file_name).as_default() as graph:

        ground_truth_input = tf.placeholder(
            tf.float32, [None, 2], name='GroundTruthInput')
        
        image_buffer_input = graph.get_tensor_by_name('input:0')
        final_tensor = graph.get_tensor_by_name('final_result:0')
        accuracy, prediction = retrain.add_evaluation_step(final_tensor, ground_truth_input)
        
        logits = graph.get_tensor_by_name("final_training_ops/W2_plus_b2/add:0")
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels = ground_truth_input,
            logits = logits))
        
    image_dir = 'tf_files/images/validation'
    testing_percentage = 100
    validation_percentage = 0
    category='testing'
    
    image_lists = retrain.create_image_lists(
        image_dir, testing_percentage,
        validation_percentage)

    class_count = len(image_lists.keys())
    
    ground_truths = []
    filenames = []
    y_true = []

    print(image_lists.keys())

    data = {}
    data['label_outros'] = int(image_lists.keys()[0] == 'melanoma')
    data['label_melanoma'] = int(image_lists.keys()[1] == 'melanoma')

    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(image_lists[label_name][category]):
        image_name = retrain.get_image_path(
            image_lists, label_name, image_index, image_dir, category)

        ground_truth = np.zeros([1, class_count], dtype=np.float32)
        ground_truth[0, label_index] = 1.0
        ground_truths.append(ground_truth)
        y_true.append(label_index)
        filenames.append(image_name)
    
    accuracies = []
    xents = []
    y = []
    y_scores = []

    with tf.Session(graph=graph) as sess:

        #print('=== MISCLASSIFIED TEST IMAGES ===')

        for filename, ground_truth in zip(filenames, ground_truths):

            image = Image.open(filename).resize((224,224),Image.ANTIALIAS)
            image = np.array(image, dtype=np.float32)[None,...]
            image = (image-127)/127.0

            feed_dict={
                image_buffer_input: image,
                ground_truth_input: ground_truth}

            eval_accuracy, eval_xent = sess.run([accuracy, xent], feed_dict)
            eval_prediction = sess.run([prediction], feed_dict)
            eval_softmax = sess.run([final_tensor], feed_dict)

            #if eval_prediction != ground_truth.argmax():    
            #    print('%s' %
            #            (filename))
            
            accuracies.append(eval_accuracy)
            xents.append(eval_xent)

            y.append(eval_prediction[0][0])
            y_scores.append(eval_softmax[0][0][1])


    data['y_true'] = y_true
    data['y_pred'] = y

    print(data)

    return data

def save_data_as_json(data):
    data_json = json.dumps(data)
    with open(sys.argv[2], "w") as json_file:
        json_file.write(data_json)

if __name__ == "__main__":
    data = evaluate_graph(sys.argv[1])
    save_data_as_json(data)
