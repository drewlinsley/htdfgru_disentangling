import os
import numpy as np
import tensorflow as tf
from ops import data_loader
from utils import py_utils
from config import Config
from tqdm import tqdm


def extract_dataset(dataset, config, cv, out_dir):
    """Save dataset npys into a directory"""
    dataset_module = py_utils.import_module(
        pre_path=config.dataset_classes,
        module=dataset)
    dataset_module = dataset_module.data_processing()
    (   
        train_data,
        _, 
        _) = py_utils.get_data_pointers(
            dataset=dataset_module.output_name,
            base_dir="/media/data_cifs/cluttered_nist_experiments/tf_records",  # config.tf_records,
            local_dir="/media/data_cifs/cluttered_nist_experiments/tf_records",  # config.local_tf_records,
            cv=cv)
    train_images, train_labels, train_aux = data_loader.inputs(
        dataset=train_data,
        batch_size=1000,  # config.train_batch_size,
        model_input_image_size=dataset_module.model_input_image_size,
        tf_dict=dataset_module.tf_dict,
        data_augmentations=[],  # config.train_augmentations,
        num_epochs=1,  # config.epochs,
        aux=None,  # train_aux_loss,
        tf_reader_settings=dataset_module.tf_reader,
        shuffle=False)  # config.shuffle_train)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(
        tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    count = 0
    try:
        while not coord.should_stop():
            images, labels = sess.run([train_images, train_labels])
            np.savez(os.path.join(out_dir, "{}".format(count)), images=images, labels=labels)
            count += 1
    except tf.errors.OutOfRangeError:
        print("Finished loop")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


def main():
    # datasets = ["curv_contour_length_6_full", "curv_contour_length_9_full", "curv_contour_length_14_full"]
    datasets = ["curv_contour_length_14_full"]  # "curv_contour_length_6_full", "curv_contour_length_9_full", "curv_contour_length_14_full"]
    cvs = ["val", "train"]
    config = Config()
    for ds in datasets:    
        for cv in cvs:
            py_utils.make_dir(ds)
            out_dir = os.path.join(ds, cv)
            py_utils.make_dir(out_dir)
            extract_dataset(dataset=ds, config=config, cv=cv, out_dir=out_dir)


if __name__ == '__main__':
    main()

