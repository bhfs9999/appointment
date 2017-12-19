#! /usr/bin/env python

import codecs
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from collections import defaultdict
import csv
import pickle as pkl
import pandas as pd

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_file", "../../../data/cnn_data/data_for_cnn_test.txt", "Data source")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.data_file)
    y_test = np.argmax(y_test, axis=1)

else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))

A = defaultdict(int) #预测正确的各个类的数目
B = defaultdict(int) #测试数据集中各个类的数目
C = defaultdict(int) #预测结果中各个类的数目
depart_dict = pkl.load(open('../../../data/cnn_data/cnn_department.pkl', 'rb'), encoding='utf-8')
depart_dict = dict(zip(depart_dict.values(), depart_dict.keys()))

for i in range(len(predictions_human_readable)):
    predict = int(float(predictions_human_readable[i][1]))
    real = int(y_test[i])
    B[real] += 1
    C[predict] += 1
    if real == predict:
        A[real] += 1

df = pd.DataFrame(index = depart_dict.values(), columns=['precision','recall','f1-measure'])

for key in B:
    r = float(A[key]) / float(B[key])
    p = float(A[key]) / float(C[key])
    f1 = p * r * 2 / (p + r)
    print(" %-16s\t p:%f\t r:%f\t f:%f\t" % (depart_dict[key], p, r, f1))
    df.loc[depart_dict[key]] = [p, r, f1]

df.to_csv("result_using_cnn.csv", encoding = "utf-8")

out_path = os.path.join(FLAGS.checkpoint_dir, "..", "result_using_cnn.csv")
print("Saving evaluation to {0}".format(out_path))
df.to_csv(out_path, encoding = "utf-8")
