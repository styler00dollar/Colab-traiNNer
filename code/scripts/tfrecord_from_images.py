"""
28-Okt-21
https://github.com/zsyzzsoft/co-mod-gan/blob/master/dataset_tools/create_from_images.py
https://github.com/zsyzzsoft/co-mod-gan/blob/master/dataset_tools/tfrecord_utils.py
"""

# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Tool for creating TFRecords datasets."""

import os
import numpy as np
import tensorflow as tf

# ----------------------------------------------------------------------------


class TFRecordExporter:
    def __init__(self, tfrecord_dir, compressed=False):
        self.tfrecord_dir = tfrecord_dir
        self.num_val_images = 0
        self.tfr_prefix = os.path.join(
            self.tfrecord_dir, os.path.basename(self.tfrecord_dir)
        )
        self.shape = None
        self.resolution_log2 = None
        self.tfr_writer = None
        self.compressed = compressed

        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        self.tfr_writer.close()
        self.tfr_writer = None

    def set_shape(self, shape):
        self.shape = shape
        self.resolution_log2 = int(np.log2(self.shape[1]))
        tfr_opt = tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.NONE
        )
        tfr_file = self.tfr_prefix + "-r%02d.tfrecords" % self.resolution_log2
        self.tfr_writer = tf.python_io.TFRecordWriter(tfr_file, tfr_opt)

    def set_num_val_images(self, num_val_images):
        self.num_val_images = num_val_images

    def add_image(self, img):
        if self.shape is None:
            self.set_shape(img.shape)
        if not self.compressed:
            assert list(self.shape) == list(img.shape)
        quant = (
            np.rint(img).clip(0, 255).astype(np.uint8) if not self.compressed else img
        )
        ex = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "shape": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=quant.shape)
                    ),
                    "data": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[quant.tostring()])
                    ),
                    "compressed": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[self.compressed])
                    ),
                    "num_val_images": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[self.num_val_images])
                    ),
                }
            )
        )
        self.tfr_writer.write(ex.SerializeToString())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


import multiprocessing as mp
import numpy as np
import argparse
from tqdm import tqdm
import random
import os
import PIL.Image

# from tfrecord_utils import TFRecordExporter


def worker(in_queue, out_queue, resolution, compressed, pix2pix):
    while True:
        fpath = in_queue.get()
        if compressed:
            assert not pix2pix
            if fpath.endswith(".avif") or fpath.endswith(".WEBP"):
                img = np.fromfile(fpath, dtype="uint8")
            else:
                img = None
        else:
            try:
                img = PIL.Image.open(fpath)
            except IOError:
                img = None
            else:
                """
                img_size = min(img.size[0] // 2 if pix2pix else img.size[1], img.size[1])
                left = (img.size[0] - (img_size * 2 if pix2pix else img_size)) // 2
                top = (img.size[1] - img_size) // 2
                img = img.crop((left, top, left + (img_size * 2 if pix2pix else img_size), top + img_size))
                img = img.resize((resolution * 2 if pix2pix else resolution, resolution), PIL.Image.BILINEAR)
                """
                img = img.resize((resolution, resolution), PIL.Image.BILINEAR)
                img = np.asarray(img.convert("RGB")).transpose([2, 0, 1])
                if pix2pix:
                    img = np.concatenate(np.split(img, 2, axis=-1), axis=0)
        out_queue.put(img)


def create_from_images(
    tfrecord_dir,
    val_image_dir,
    train_image_dir,
    resolution,
    num_channels,
    num_processes,
    shuffle,
    compressed,
    pix2pix,
):
    in_queue = mp.Queue()
    out_queue = mp.Queue(num_processes * 8)

    worker_procs = [
        mp.Process(
            target=worker, args=(in_queue, out_queue, resolution, compressed, pix2pix)
        )
        for _ in range(num_processes)
    ]
    for worker_proc in worker_procs:
        worker_proc.daemon = True
        worker_proc.start()

    print("Processes created.")

    with TFRecordExporter(tfrecord_dir, compressed=compressed) as tfr:
        tfr.set_shape(
            [num_channels * 2 if pix2pix else num_channels, resolution, resolution]
        )

        if val_image_dir:
            print("Processing validation images...")
            flist = []
            for root, _, files in os.walk(val_image_dir):
                print(root)
                flist.extend([os.path.join(root, fname) for fname in files])
            tfr.set_num_val_images(len(flist))
            if shuffle:
                random.shuffle(flist)
            for fpath in tqdm(flist):
                in_queue.put(fpath, block=False)
            for _ in tqdm(range(len(flist))):
                img = out_queue.get()
                if img is not None:
                    tfr.add_image(img)

        if train_image_dir:
            print("Processing training images...")
            flist = []
            for root, _, files in os.walk(train_image_dir):
                print(root)
                flist.extend([os.path.join(root, fname) for fname in files])
            if shuffle:
                random.shuffle(flist)
            for fpath in tqdm(flist):
                in_queue.put(fpath, block=False)
            for _ in tqdm(range(len(flist))):
                img = out_queue.get()
                if img is not None:
                    tfr.add_image(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tfrecord-dir", help="Output directory of generated TFRecord", required=True
    )
    parser.add_argument(
        "--val-image-dir", help="Root directory of validation images", default=None
    )
    parser.add_argument(
        "--train-image-dir", help="Root directory of training images", default=None
    )
    parser.add_argument("--resolution", help="Target resolution", type=int, default=512)
    parser.add_argument(
        "--num-channels", help="Number of channels of images", type=int, default=3
    )
    parser.add_argument(
        "--num-processes", help="Number of parallel processes", type=int, default=8
    )
    parser.add_argument("--shuffle", default=False, action="store_true")
    parser.add_argument("--compressed", default=False, action="store_true")
    parser.add_argument("--pix2pix", default=False, action="store_true")

    args = parser.parse_args()
    create_from_images(**vars(args))


if __name__ == "__main__":
    main()
