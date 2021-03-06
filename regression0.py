# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow import keras

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


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=[8]),
        keras.layers.Dense(1),
    ])

    model.compile(loss="mean_squared_error", optimizer="sgd")

    print(model.summary(), flush=True)

    return model


def run():
    tfr_train_filenames = ["./data/tfrecords/house/train.tfrecords"]
    tfr_valid_filenames = ["./data/tfrecords/house/valid.tfrecords"]
    tfr_test_filenames = ["./data/tfrecords/house/test.tfrecords"]

    train_dataset = tfrecords_reader_dataset(tfr_train_filenames, n_readers=1, compression_type="GZIP")
    valid_dataset = tfrecords_reader_dataset(tfr_valid_filenames, n_readers=1, compression_type="GZIP")
    test_dataset = tfrecords_reader_dataset(tfr_test_filenames, n_readers=1, compression_type="GZIP")
    model = build_model()

    print("train" + "=" * 10)
    output_model_file = "./output/model/house/h5/model.h5"
    logdir = "./output/tensorboard/house"
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2),
        keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True, save_weight_only=False),
        keras.callbacks.TensorBoard(logdir)
    ]
    history = model.fit(train_dataset,
                        validation_data=valid_dataset,
                        steps_per_epoch=11160 // 32,
                        validation_steps=3870 // 32,
                        epochs=100,
                        callbacks=callbacks)

    print(history.history)

    print("test" + "=" * 10)
    model.evaluate(test_dataset, steps=5160 // 32)


def run2():
    tfr_train_filenames = ["./data/tfrecords/house/train.tfrecords"]
    tfr_valid_filenames = ["./data/tfrecords/house/valid.tfrecords"]
    tfr_test_filenames = ["./data/tfrecords/house/test.tfrecords"]

    train_dataset = tfrecords_reader_dataset(tfr_train_filenames, n_readers=1, compression_type="GZIP")
    valid_dataset = tfrecords_reader_dataset(tfr_valid_filenames, n_readers=1, compression_type="GZIP")
    test_dataset = tfrecords_reader_dataset(tfr_test_filenames, n_readers=1, compression_type="GZIP")
    model = build_model()

    print("train" + "=" * 10)
    logdir = "./output/tensorboard/house"
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2),
        keras.callbacks.TensorBoard(logdir)
    ]
    history = model.fit(train_dataset,
                        validation_data=valid_dataset,
                        steps_per_epoch=11160 // 32,
                        validation_steps=3870 // 32,
                        epochs=100,
                        callbacks=callbacks)

    print(history.history)

    print("test" + "=" * 10)
    model.evaluate(test_dataset, steps=5160 // 32)

    output_model = "./output/model/house/saved_graph/"
    tf.saved_model.save(model, output_model)


def load_saved_model():
    output_model = "./output/model/house/saved_graph/"
    model = tf.saved_model.load(output_model)
    print(model.signatures.keys())
    inference = model.signatures["serving_default"]
    print(inference)
    print(inference.structured_outputs)
    x = tf.constant([[1.0] * 8], dtype=tf.float32)
    y = inference(x)

    print(y)
    print(y["dense_1"])


if __name__ == '__main__':
    load_saved_model()
