from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
# 'yolov8n.pt' is the smallest and fastest model. Other options:
# yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
img_path = "Stop-Sign.v1i.tensorflow/valid/00014_00001_00025_png_jpg.rf.9b82bfc941ffc9127ad9c219516e4e27.jpg"
model = YOLO('yolov8n.pt')

# Run inference on an image
# You can use a path to an image, a URL, or even a YouTube video URL.
results = model(img_path)

# Process results
# The 'results' object contains the detections.
# You can view, print, or save the results.

# Option 1: Display the image with detections
results[0].show()

# Option 2: Save the image with detections
results[0].save(filename='result.jpg') 