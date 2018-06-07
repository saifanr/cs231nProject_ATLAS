from __future__ import division
import os
import numpy as np
import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle, csv

from utils import *
from model import UNet3D

flags = tf.app.flags
flags.DEFINE_integer("epoch", 4, "Epoch to train [4]")
flags.DEFINE_string("train_patch_dir", "../train", "Directory of the training data patches")
flags.DEFINE_bool("split_train", False, "Whether to split the train data into train and val")
flags.DEFINE_string("train_data_dir", "../train", "Directory of the train data")
flags.DEFINE_string("testing_data_dir", None, "Directory of the testing data")
flags.DEFINE_string("deploy_data_dir", "../test", "Directory of the test data")
flags.DEFINE_string("deploy_output_dir", "output_validation", "Directory name of the output data")
flags.DEFINE_integer("batch_size", 4, "Batch size")
flags.DEFINE_integer("seg_features_root", 48, "Number of features in the first filter in the seg net [48]")
flags.DEFINE_integer("conv_size", 3, "Convolution kernel size in encoding and decoding paths [3]")
flags.DEFINE_integer("layers", 3, "Encoding and deconding layers [3]")
flags.DEFINE_string("loss_type", "cross_entropy", "Loss type in the model [cross_entropy]")
flags.DEFINE_float("dropout", 0.5, "Drop out ratio [0.5]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "logs", "Directory name to save logs [logs]")
flags.DEFINE_boolean("train", False, "True for training, False for deploying [False]")
flags.DEFINE_boolean("run_seg", True, "True if run segmentation [True]")
FLAGS = flags.FLAGS

def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    # Train
    if FLAGS.split_train:
        if os.path.exists(os.path.join(FLAGS.train_patch_dir, 'files.log')):
            with open(os.path.join(FLAGS.train_patch_dir, 'files.log'), 'r') as f:
                training_paths, testing_paths = pickle.load(f)
        else:
            all_paths = [os.path.join(FLAGS.train_patch_dir, p) for p in sorted(os.listdir(FLAGS.train_data_dir))]
            np.random.shuffle(all_paths)
            n_training = int(len(all_paths) * 4 / 5)
            training_paths = all_paths[:n_training]
            testing_paths = all_paths[n_training:]
            # Save the training paths and testing paths
            with open(os.path.join(FLAGS.train_data_dir, 'files.log'), 'w') as f:
                pickle.dump([training_paths, testing_paths], f)

        training_ids = [os.path.basename(i) for i in training_paths]
        testing_ids = [os.path.basename(i) for i in testing_paths]

    else:
        # train_patch_dir = data/ATLAS_R1.1/train/
        training_paths = []
        for dirpath, dirnames, files in os.walk(FLAGS.train_patch_dir):
            if os.path.basename(dirpath)[0:7] == 'patches':
                training_paths.append(dirpath)

        testing_paths = []
        for dirpath, dirnames, files in os.walk(FLAGS.testing_data_dir):
            if os.path.basename(dirpath)[0:7] == 'patches':
                testing_paths.append(dirpath)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    # Segmentation net
    if FLAGS.run_seg:
        run_config = tf.ConfigProto()
        with tf.Session(config=run_config) as sess:
            unet = UNet3D(sess, checkpoint_dir=FLAGS.checkpoint_dir, log_dir=FLAGS.log_dir, training_paths=training_paths,
                          testing_paths=testing_paths, batch_size=FLAGS.batch_size, layers=FLAGS.layers,
                          features_root=FLAGS.seg_features_root, conv_size=FLAGS.conv_size,
                          dropout=FLAGS.dropout, loss_type=FLAGS.loss_type)

            if FLAGS.train:
                model_vars = tf.trainable_variables()
                slim.model_analyzer.analyze_vars(model_vars, print_info=True)

                train_config = {}
                train_config['epoch'] = FLAGS.epoch

                unet.train(train_config)
            else:
                # Deploy
                unet.deploy(FLAGS.deploy_data_dir)

        tf.reset_default_graph()

if __name__ == '__main__':
    tf.app.run()
