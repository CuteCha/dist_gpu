# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf


def parse_csv_line(line, n_fields=9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x, y


def csv_reader_dataset(filenames, n_readers=5, batch_size=32, n_parse_threads=5, shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename),
        cycle_length=n_readers)
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset


def serialize_example(x, y):
    """Converts x,y to tf.train.Example and serialize"""
    input_features = tf.train.FloatList(value=x)
    label = tf.train.FloatList(value=y)
    features = tf.train.Features(
        feature={
            "input_features": tf.train.Feature(float_list=input_features),
            "label": tf.train.Feature(float_list=label)
        }
    )
    example = tf.train.Example(features=features)
    return example.SerializeToString()


def csv_to_tfrecords(csv_filename, tfrecords_filename, compression_type=None):
    options = tf.io.TFRecordOptions(compression_type=compression_type)
    data = np.loadtxt(csv_filename, delimiter=',')

    with tf.io.TFRecordWriter(tfrecords_filename, options) as writer:
        for line in data:
            x = line[:-1]
            y = line[-1:]
            writer.write(serialize_example(x, y))


expected_features = {
    "input_features": tf.io.FixedLenFeature([8], dtype=tf.float32),
    "label": tf.io.FixedLenFeature([1], dtype=tf.float32)
}


def parse_example(serialized_example):
    example = tf.io.parse_single_example(serialized_example, expected_features)
    return example["input_features"], example["label"]


def tfrecords_reader_dataset(filenames, n_readers=5, batch_size=32, n_parse_threads=5,
                             shuffle_buffer_size=10000, compression_type=None):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(
            filename, compression_type=compression_type),
        cycle_length=n_readers)
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_example, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset
