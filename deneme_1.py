import argparse
import logging
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import pyzed.sl as sl

run_signal = False
exit_signal = False
detections = []
i = 0

logging.basicConfig(level=logging.INFO)

def load_model():
    """
    Loads the pre-trained YOLO model.
    """
    model = YOLO('yolov10m.pt')  # medium-sized model for balance of speed and accuracy
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model

def measure_box(zed, bbox):
    """
    Calculates the width, height, and depth of the detected box using ZED camera.
    Uses multiple depth points within the bounding box for more accurate measurements.
    """
    x1, y1, x2, y2 = map(int, bbox)  # Ensure integers for indexing
    depth_values = []

    # Get camera resolution
    cam_info = zed.get_camera_information()
    width = cam_info.camera_configuration.resolution.width
    height = cam_info.camera_configuration.resolution.height

    # Sample multiple points within the bounding box for depth values
    step = max(1, min((x2 - x1) // 10, (y2 - y1) // 10))  # Adaptive sampling density
    for x in range(x1, x2, step):
        for y in range(y1, y2, step):
            if 0 <= y < height and 0 <= x < width:  # Fixed attribute names
                depth_value = sl.Mat()
                zed.retrieve_measure(depth_value, sl.MEASURE.DEPTH)
                depth_image = depth_value.get_data()
                depth = depth_image[y, x]
                if depth > 0 and depth < 1000:  # Filter out invalid and extreme depth values
                    depth_values.append(depth)

    if depth_values:
        # Use the median of depth values to reduce the effect of outliers
        median_depth = sorted(depth_values)[len(depth_values) // 2]
        
        # Get camera parameters for more accurate measurements
        fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
        
        # Calculate real-world dimensions using perspective projection
        width = abs(x2 - x1) * median_depth / fx
        height = abs(y2 - y1) * median_depth / fx
        return width, height, median_depth
    return None, None, None

def detect_objects(model, frame, conf_threshold=0.5):
    """
    Detects objects in the given frame using the YOLO model.
    """
    # Convert RGBA to RGB
    if frame.shape[2] == 4:  # If image has 4 channels (RGBA)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    
    results = model(frame, conf=conf_threshold)
    rects = []
    
    # Process results - YOLOv8 format
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            rects.append([int(x1), int(y1), int(x2), int(y2)])
    
    return rects

def draw_rectangle_with_text(frame, bbox, text):
    """
    Draws a rectangle and text on the frame with shadows for better readability.
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add background for text
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_w, y1), (0, 255, 0), -1)
    
    # Add text with shadow for better visibility
    cv2.putText(frame, text, (x1 + 1, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def main():
    global exit_signal, run_signal, i, detections
    
    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.CENTIMETER
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Set to 720p for better performance
    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        logging.error(f"Camera Open : {repr(status)}. Exit program.")
        return

    try:
        # Load YOLO model
        model = load_model()
        logging.info("Model loaded successfully")

        runtime_params = sl.RuntimeParameters()
        image = sl.Mat()
        depth = sl.Mat()

        logging.info("Object detection: Running...")
        
        fps_start_time = time.time()
        fps = 0
        frame_count = 0

        while not exit_signal:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # FPS calculation
                frame_count += 1
                if frame_count >= 30:  # Update FPS every 30 frames
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    frame_count = 0

                # Retrieve images
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()

                # Object detection
                rects = detect_objects(model, frame)
                detections = rects

                # Process detections
                for bbox in detections:
                    width, height, depth_val = measure_box(zed, bbox)
                    if width and height and depth_val:
                        text = f"W: {width:.1f}cm, H: {height:.1f}cm, D: {depth_val:.1f}cm"
                        draw_rectangle_with_text(frame, bbox, text)

                # Display FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Convert to RGB for display
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                cv2.imshow("ZED | 2D View", display_frame)

                key = cv2.waitKey(1)
                if key == ord("s"):  # 's' to measure and save
                    if detections:
                        for bbox in detections:
                            i += 1
                            width, height, depth_val = measure_box(zed, bbox)
                            if width and height and depth_val:
                                with open("box_measurements.txt", "a") as f:
                                    f.write(f"Box{i} dimensions: Width={width:.2f}cm, Height={height:.2f}cm, Depth={depth_val:.2f}cm\n")
                                logging.info(f"Box{i} measured and saved: {width:.2f}cm x {height:.2f}cm x {depth_val:.2f}cm")
                elif key in (27, ord("q")):  # ESC or 'q' to exit
                    exit_signal = True

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    opt = parser.parse_args()
    main()