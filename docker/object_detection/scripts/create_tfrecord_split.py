#!/usr/bin/env python
# coding: utf-8

"""
TFRecord creation script with automatic 80/20 train/val split
"""

import os
import io
import tensorflow as tf
import PIL.Image
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util
import glob
import random
import numpy as np

# Configure logging
tf.get_logger().setLevel('INFO')

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

flags = tf.compat.v1.app.flags
flags.DEFINE_string('output_dir', '', 'Path to output directory for TFRecords')
flags.DEFINE_string('image_dir', '', 'Path to directory containing images')
flags.DEFINE_string('annotations_dir', '', 'Path to directory containing Pascal VOC XML files')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
flags.DEFINE_float('val_split', 0.2, 'Fraction of data to use for validation (default: 0.2)')
FLAGS = flags.FLAGS

def get_label_map_dict(label_map_path):
    """Reads label map and returns a dictionary mapping names to ids."""
    from object_detection.protos import string_int_label_map_pb2
    from google.protobuf import text_format

    try:
        with tf.io.gfile.GFile(label_map_path, 'r') as fid:
            label_map_string = fid.read()
            label_map = string_int_label_map_pb2.StringIntLabelMap()
            text_format.Merge(label_map_string, label_map)

        label_map_dict = {}
        for item in label_map.item:
            label_map_dict[item.name] = item.id
        return label_map_dict
    except Exception as e:
        print(f"Error loading label map: {e}")
        return None

def xml_to_dict(xml_file):
    """Convert Pascal VOC XML to dictionary."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find('filename').text
        if filename is None:
            print(f"Warning: No filename found in {xml_file}")
            return None

        xml_dict = {
            'filename': filename,
            'size': {
                'width': int(root.find('size/width').text),
                'height': int(root.find('size/height').text),
                'depth': int(root.find('size/depth').text)
            },
            'objects': []
        }

        for obj in root.findall('object'):
            try:
                obj_dict = {
                    'name': obj.find('name').text,
                    'bndbox': {
                        'xmin': float(obj.find('bndbox/xmin').text),
                        'ymin': float(obj.find('bndbox/ymin').text),
                        'xmax': float(obj.find('bndbox/xmax').text),
                        'ymax': float(obj.find('bndbox/ymax').text)
                    }
                }
                xml_dict['objects'].append(obj_dict)
            except Exception as e:
                print(f"Warning: Error parsing object in {xml_file}: {e}")
                continue

        return xml_dict
    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return None

def create_tf_example(data, image_dir, label_map_dict):
    """Convert XML derived dict to tf.Example proto."""
    try:
        image_path = os.path.join(image_dir, data['filename'])
        if not tf.io.gfile.exists(image_path):
            print(f"Image not found: {image_path}")
            return None

        with tf.io.gfile.GFile(image_path, 'rb') as fid:
            encoded_jpg = fid.read()

        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)

        width = data['size']['width']
        height = data['size']['height']

        filename = data['filename'].encode('utf8')
        image_format = b'jpg'

        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for obj in data['objects']:
            if obj['name'] not in label_map_dict:
                print(f"Warning: Class {obj['name']} not in label map, skipping")
                continue

            xmins.append(obj['bndbox']['xmin'] / width)
            xmaxs.append(obj['bndbox']['xmax'] / width)
            ymins.append(obj['bndbox']['ymin'] / height)
            ymaxs.append(obj['bndbox']['ymax'] / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])

        if not xmins:  # Skip if no valid objects found
            print(f"Warning: No valid objects found in {data['filename']}")
            return None

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))

        return tf_example
    except Exception as e:
        print(f"Error creating TF example for {data.get('filename', 'unknown')}: {e}")
        return None

def split_dataset(xml_files, val_split=0.2):
    """Randomly split dataset into train and validation sets."""
    random.shuffle(xml_files)
    split_idx = int(len(xml_files) * (1 - val_split))
    return xml_files[:split_idx], xml_files[split_idx:]

def write_tfrecord(writer, xml_files, image_dir, label_map_dict):
    """Write examples to TFRecord file."""
    num_success = 0
    for xml_file in xml_files:
        data = xml_to_dict(xml_file)
        if data is None:
            continue

        tf_example = create_tf_example(data, image_dir, label_map_dict)
        if tf_example is None:
            continue

        writer.write(tf_example.SerializeToString())
        num_success += 1

        if num_success % 100 == 0:
            print(f"Processed {num_success} examples...")
    return num_success

def main(_):
    print("\n=== Starting TFRecord creation with train/val split ===")
    print(f"Output directory: {FLAGS.output_dir}")
    print(f"Image dir: {FLAGS.image_dir}")
    print(f"Annotations dir: {FLAGS.annotations_dir}")
    print(f"Label map path: {FLAGS.label_map_path}")
    print(f"Validation split: {FLAGS.val_split}\n")

    # Verify inputs exist
    if not all([FLAGS.output_dir, FLAGS.image_dir, FLAGS.annotations_dir, FLAGS.label_map_path]):
        print("Error: All flags must be specified")
        return

    if not tf.io.gfile.exists(FLAGS.image_dir):
        print(f"Error: Image directory not found: {FLAGS.image_dir}")
        return

    if not tf.io.gfile.exists(FLAGS.annotations_dir):
        print(f"Error: Annotations directory not found: {FLAGS.annotations_dir}")
        return

    if not tf.io.gfile.exists(FLAGS.label_map_path):
        print(f"Error: Label map file not found: {FLAGS.label_map_path}")
        return

    # Load label map
    label_map_dict = get_label_map_dict(FLAGS.label_map_path)
    if label_map_dict is None:
        print("Error: Failed to load label map")
        return

    print(f"Loaded label map with {len(label_map_dict)} classes")

    # Get XML files and split
    xml_files = glob.glob(os.path.join(FLAGS.annotations_dir, '*.xml'))
    if not xml_files:
        print(f"Error: No XML files found in {FLAGS.annotations_dir}")
        return

    train_files, val_files = split_dataset(xml_files, FLAGS.val_split)
    print(f"Found {len(xml_files)} total examples")
    print(f"Splitting into {len(train_files)} training and {len(val_files)} validation examples")

    # Create output directory if it doesn't exist
    if not tf.io.gfile.exists(FLAGS.output_dir):
        tf.io.gfile.makedirs(FLAGS.output_dir)

    # Create train.record
    train_path = os.path.join(FLAGS.output_dir, 'train.record')
    print(f"\nCreating training set: {train_path}")
    with tf.io.TFRecordWriter(train_path) as writer:
        train_success = write_tfrecord(writer, train_files, FLAGS.image_dir, label_map_dict)

    # Create val.record
    val_path = os.path.join(FLAGS.output_dir, 'val.record')
    print(f"\nCreating validation set: {val_path}")
    with tf.io.TFRecordWriter(val_path) as writer:
        val_success = write_tfrecord(writer, val_files, FLAGS.image_dir, label_map_dict)

    # Final report
    print("\n=== Final Report ===")
    print(f"Training examples: {train_success}/{len(train_files)}")
    print(f"Validation examples: {val_success}/{len(val_files)}")
    print(f"Total processed: {train_success + val_success}/{len(xml_files)}")
    print(f"Training TFRecord: {train_path}")
    print(f"Validation TFRecord: {val_path}")

    # Verify output files
    if train_success > 0:
        train_size = os.path.getsize(train_path)
        print(f"Training file size: {train_size} bytes")
    else:
        print("Warning: No training examples were written")

    if val_success > 0:
        val_size = os.path.getsize(val_path)
        print(f"Validation file size: {val_size} bytes")
    else:
        print("Warning: No validation examples were written")

if __name__ == '__main__':
    tf.compat.v1.app.run()
