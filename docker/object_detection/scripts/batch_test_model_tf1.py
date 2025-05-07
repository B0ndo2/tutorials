import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
import os

# Force matplotlib to use 'Agg' backend (non-interactive)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load image using PIL"""
    try:
        return np.array(Image.open(image_path).convert('RGB'))
    except Exception as e:
        print(f"Image load error: {e}")
        return None

def run_inference(image_np, detection_graph):
    """Run object detection"""
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            ops = detection_graph.get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)

            output_dict = sess.run(tensor_dict,
                                 feed_dict={'image_tensor:0': np.expand_dims(image_np, axis=0)})

            return (output_dict['detection_boxes'][0],
                   output_dict['detection_scores'][0],
                   output_dict['detection_classes'][0].astype(np.int32))

def save_visualization(image_np, boxes, scores, classes, label_map, output_path):
    """Save detection results to file"""
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    ax = plt.gca()

    height, width = image_np.shape[:2]

    for i in range(min(20, len(scores))):
        if scores[i] > 0.5:  # Confidence threshold
            box = boxes[i]
            y1, x1, y2, x2 = box
            y1, x1, y2, x2 = int(y1*height), int(x1*width), int(y2*height), int(x2*width)

            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, color='red', linewidth=2)
            ax.add_patch(rect)

            label = f"{label_map.get(classes[i], str(classes[i]))}: {scores[i]:.2f}"
            plt.text(x1, y1-10, label, color='red', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved results to {output_path}")

def process_single_image(image_path, output_dir, detection_graph, label_map):
    """Process and save results for one image"""
    image_np = load_image(image_path)
    if image_np is None:
        return

    # Create output filename
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"detected_{base_name}")

    # Run detection
    boxes, scores, classes = run_inference(image_np, detection_graph)

    # Save results
    save_visualization(image_np, boxes, scores, classes, label_map, output_path)

def main():
    # Configuration
    MODEL_PATH = 'learn_pet/models/saved_model_640_4963/frozen_inference_graph.pb'
    INPUT_DIR = 'learn_pet/pet/images'  # Directory containing images to process
    OUTPUT_DIR = 'learn_pet/eval'  # Where to save results
    LABEL_MAP = {1: 'person', 2: 'car'}  # Update with your classes

    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model once
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(MODEL_PATH, 'rb') as fid:
            od_graph_def.ParseFromString(fid.read())
            tf.import_graph_def(od_graph_def, name='')

    # Process all images in input directory
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    processed_count = 0

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(supported_extensions):
            image_path = os.path.join(INPUT_DIR, filename)
            process_single_image(image_path, OUTPUT_DIR, detection_graph, LABEL_MAP)
            processed_count += 1

    print(f"\nProcessing complete. Processed {processed_count} images.")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
