import tensorflow as tf
tf.compat.v1.enable_eager_execution()  # Forces eager mode in TF 1.x
from collections import defaultdict

# Define the feature description (should match how TFRecords were created)
feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),  # Class names
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),  # Class IDs
}

def parse_tfrecord(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

# Count class occurrences
class_counts = defaultdict(int)

# Path to your TFRecord file(s)
tfrecord_paths = ["/tensorflow/models/research/learn_pet/pet/train.record"]

# Read and parse TFRecord
raw_dataset = tf.data.TFRecordDataset(tfrecord_paths)
parsed_dataset = raw_dataset.map(parse_tfrecord)

# Iterate through records and count classes
for record in parsed_dataset:
    class_texts = record['image/object/class/text'].values.numpy()  # Get class names
    for class_text in class_texts:
        class_name = class_text.decode('utf-8')  # Convert bytes to string
        class_counts[class_name] += 1

# Print results
print("Class distribution in TFRecords:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} objects")
