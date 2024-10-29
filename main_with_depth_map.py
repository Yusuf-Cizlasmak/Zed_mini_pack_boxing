import argparse
import logging
import time

import cv2
import numpy as np
import pyzed.sl as sl
import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import torch.nn.functional as F

# Sinyal işaretleri
run_signal = False
exit_signal = False

# Tespit edilen kutuların koordinatlarını ve ölçümleri saklamak için liste
detections = []

# Her bir kutu için sıra numarası
i = 0

# Loglama seviyesini ayarla
logging.basicConfig(level=logging.INFO)

def load_models():
    """
    YOLO ve Depth Anything modellerini yükler
    """
    # YOLO modelini yükle
    yolo_model = YOLO("yolov10m.pt")
    yolo_model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Depth Anything modelini yükle
    depth_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
    depth_model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    return yolo_model, depth_model, depth_processor

def get_depth_anything_prediction(frame, depth_model, depth_processor):
    """
    Depth Anything modeli ile derinlik tahmini yapar ve orijinal görüntü boyutuna ölçekler
    """
    # BGR'dan RGB'ye dönüştür
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Görüntüyü işle
    inputs = depth_processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(depth_model.device) for k, v in inputs.items()}
    
    # Orijinal görüntü boyutlarını kaydet
    original_height, original_width = frame.shape[:2]
    
    # Derinlik tahmini yap
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth
        
        # Depth map'i orijinal görüntü boyutuna ölçekle
        predicted_depth = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(original_height, original_width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Derinlik haritasını numpy dizisine dönüştür
    depth_map = predicted_depth.cpu().numpy()
    
    # Normalize et (görselleştirme için)
    depth_map = ((depth_map - depth_map.min()) * 255 / 
                 (depth_map.max() - depth_map.min())).astype(np.uint8)
    
    return depth_map

def is_rectangular_box(bbox, aspect_ratio_threshold=0.3):
    """
    Verilen sınırlayıcı kutunun bir kutu olup olmadığını belirler.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    if width == 0 or height == 0:
        return False

    aspect_ratio = min(width, height) / max(width, height)
    return aspect_ratio >= aspect_ratio_threshold

def measure_box(zed, bbox, depth_map=None):
    """
    ZED kamera ve Depth Anything kullanarak kutu ölçümlerini yapar
    """
    x1, y1, x2, y2 = map(int, bbox)
    depth_values = []
    depth_anything_values = []

    # Kamera çözünürlüğünü al
    cam_info = zed.get_camera_information()
    width = cam_info.camera_configuration.resolution.width
    height = cam_info.camera_configuration.resolution.height

    # Kutu içindeki örnekleme adımını belirle
    step = max(1, min((x2 - x1) // 10, (y2 - y1) // 10))

    # Koordinatların görüntü sınırları içinde olduğundan emin ol
    x1, x2 = max(0, x1), min(width - 1, x2)
    y1, y2 = max(0, y1), min(height - 1, y2)

    for x in range(x1, x2, step):
        for y in range(y1, y2, step):
            # ZED depth değerini al
            depth_value = sl.Mat()
            zed.retrieve_measure(depth_value, sl.MEASURE.DEPTH)
            depth_image = depth_value.get_data()
            depth = depth_image[y, x]
            
            if depth > 0 and depth < 1000:
                depth_values.append(depth)
            
            # Depth Anything değerini al (eğer varsa)
            if depth_map is not None:
                try:
                    depth_anything_val = depth_map[y, x]
                    depth_anything_values.append(depth_anything_val)
                except IndexError:
                    logging.warning(f"Index error at coordinates ({x}, {y}) for depth map")
                    continue

    if depth_values:
        median_depth_zed = sorted(depth_values)[len(depth_values) // 2]
        median_depth_anything = None
        
        if depth_anything_values:
            median_depth_anything = sorted(depth_anything_values)[len(depth_anything_values) // 2]
            # Depth Anything değerlerini ölçekle (örnek olarak)
            scale_factor = median_depth_zed / median_depth_anything if median_depth_anything > 0 else 1
            median_depth_anything = median_depth_anything * scale_factor

        # Kamera parametrelerini kullanarak genişlik ve yükseklik hesapla
        fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
        width = abs(x2 - x1) * median_depth_zed / fx
        height = abs(y2 - y1) * median_depth_zed / fx
        
        return width, height, median_depth_zed, median_depth_anything
    
    return None, None, None, None

def detect_objects(model, frame, conf_threshold=0.5):
    """
    YOLO modelini kullanarak nesneleri tespit eder
    """
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    results = model(frame, conf=conf_threshold)
    rects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls[0] in [73, 84]:  # 73: book, 84: box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                if is_rectangular_box(bbox):
                    rects.append(bbox)

    return rects

def draw_rectangle_with_text(frame, bbox, text):
    """
    Tespit edilen kutuları ve ölçümleri ekranda gösterir
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Her bir satırı ayrı ayrı yazdır
    text_lines = text.split('\n')
    y_offset = y1 - 10
    for i, line in enumerate(text_lines):
        y = y_offset - (i * 20)  # Her satır için 20 piksel yukarı
        
        # Text background
        (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y - 15), (x1 + text_w, y + 5), (0, 255, 0), -1)
        
        # Text
        cv2.putText(frame, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def main():
    global exit_signal, run_signal, i, detections

    # ZED kamera ayarları
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.CENTIMETER
    init_params.camera_resolution = sl.RESOLUTION.HD2K

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        logging.error(f"Camera Open : {repr(status)}. Exit program.")
        return

    try:
        # Modelleri yükle
        yolo_model, depth_model, depth_processor = load_models()
        logging.info("Models loaded successfully")

        runtime_params = sl.RuntimeParameters()
        image = sl.Mat()

        logging.info("Object detection: Running...")

        fps_start_time = time.time()
        fps = 0
        frame_count = 0

        while not exit_signal:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                frame_count += 1
                if frame_count >= 30:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    frame_count = 0

                # Kamera görüntüsünü al
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()

                try:
                    # Depth Anything ile derinlik tahmini yap
                    depth_map = get_depth_anything_prediction(frame, depth_model, depth_processor)

                    # Nesneleri tespit et
                    rects = detect_objects(yolo_model, frame)
                    detections = rects

                    # Kutuları çiz ve ölçümleri göster
                    for bbox in detections:
                        width, height, depth_val_zed, depth_val_anything = measure_box(zed, bbox, depth_map)
                        if width and height and depth_val_zed:
                            text = f"W: {width:.1f}cm, H: {height:.1f}cm\n"
                            text += f"ZED: {depth_val_zed:.1f}cm"
                            if depth_val_anything:
                                text += f"\nDA: {depth_val_anything:.1f}cm"
                            draw_rectangle_with_text(frame, bbox, text)

                    # FPS göster
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Depth map'i görselleştir
                    depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
                    
                    # Ana görüntüyü göster
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                    cv2.imshow("ZED | 2D View", display_frame, )
                    
                    # Depth map'i göster
                    cv2.imshow("Depth Anything Prediction", depth_map_colored, )

                except Exception as e:
                    logging.error(f"Frame processing error: {str(e)}")
                    continue

                # Klavye işlemleri
                key = cv2.waitKey(1)
                if key == ord("s"):
                    if detections:
                        for bbox in detections:
                            i += 1
                            width, height, depth_val_zed, depth_val_anything = measure_box(zed, bbox, depth_map)
                            if width and height and depth_val_zed:
                                with open("box_measurements.txt", "a") as f:
                                    measurement_text = f"Box{i} dimensions: Width={width:.2f}cm, Height={height:.2f}cm, "
                                    measurement_text += f"ZED Depth={depth_val_zed:.2f}cm"
                                    if depth_val_anything:
                                        measurement_text += f", Depth Anything={depth_val_anything:.2f}cm"
                                    measurement_text += "\n"
                                    f.write(measurement_text)
                                logging.info(f"Box{i} measured and saved")

                elif key in (27, ord("q")):
                    exit_signal = True

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.6, help="confidence threshold")
    opt = parser.parse_args()
    main()