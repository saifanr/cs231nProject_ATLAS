import os
import tensorflow as tf
import random as rn
import numpy as np

flags = tf.app.flags
flags.DEFINE_string("current_data_dir", "data", "Current Directory of the combined data")
flags.DEFINE_string("train_data_dir", "train", "Directory of the training data")
flags.DEFINE_string("test_data_dir", "test", "Directory of the test data")
FLAGS = flags.FLAGS

paths = []

if __name__ == '__main__':
    for dirpath, dirnames, files in os.walk(FLAGS.current_data_dir):
        if os.path.basename(dirpath)[0:4] == 'Site':
            for dir in dirnames:
                paths.append( os.path.join( dirpath, dir ) )

    rn.shuffle(paths)

    train_paths = paths[0:len(paths)*80//100]
    test_paths = paths[len(paths)*80//100:]

    for path in train_paths:
        if not os.path.exists( os.path.join(FLAGS.train_data_dir, os.path.relpath(path,FLAGS.current_data_dir)) ):
            os.makedirs( os.path.join(FLAGS.train_data_dir, os.path.relpath(path,FLAGS.current_data_dir)) )
        os.rename( path, os.path.join(FLAGS.train_data_dir, os.path.relpath(path,FLAGS.current_data_dir)) )

    for path in test_paths:
        if not os.path.exists( os.path.join(FLAGS.test_data_dir, os.path.relpath(path,FLAGS.current_data_dir)) ):
            os.makedirs( os.path.join(FLAGS.test_data_dir, os.path.relpath(path,FLAGS.current_data_dir)) )
        os.rename( path, os.path.join(FLAGS.test_data_dir, os.path.relpath(path,FLAGS.current_data_dir)) )
