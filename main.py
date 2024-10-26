import argparse
import logging
import time

import cv2
import numpy as np
import pyzed.sl as sl
import torch
from ultralytics import YOLO

# Sinyal işaretleri
run_signal = False
exit_signal = False


# Tespit edilen kutuların koordinatlarını ve ölçümleri saklamak için liste
detections = []

# Her bir kutu için sıra numarası
i = 0


# Loglama seviyesini ayarla
logging.basicConfig(level=logging.INFO)


def load_model():
    """
    YOLO modelini yükler ve CUDA'ya taşır. Burada 'yolov10m.pt' modeli kullanılmıştır.
    """
    model = YOLO("yolov10m.pt")  # medium-sized model for balance of speed and accuracy
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def is_rectangular_box(bbox, aspect_ratio_threshold=0.3):
    """
    Verilen sınırlayıcı kutunun bir kutu olup olmadığını belirler.

    bbox: [x1, y1, x2, y2] formatında sınırlayıcı kutu koordinatları
    aspect_ratio_threshold: kabul edilebilir bir kutu için : (genişlik/yükseklik) oranı 
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    if width == 0 or height == 0:
        return False

    aspect_ratio = min(width, height) / max(width, height)

    # Genişlik ve yükseklik oranı eşik değerinden büyükse, bir kutu olarak kabul edilir
    return aspect_ratio >= aspect_ratio_threshold


def measure_box(zed, bbox):
    """
    zed: ZED kamera nesnesi
    bbox: [x1, y1, x2, y2] formatında sınırlayıcı kutu koordinatları

    Verilen sınırlayıcı kutunun genişliğini, yüksekliğini ve derinliğini ölçer.
    #* 3B noktaları 2B piksel koordinatlarına dönüştürür ve kutu içindeki derinlik değerlerinin ortanca değerini alır.
    """
    x1, y1, x2, y2 = map(int, bbox)  # Convert to integers: (x1, y1, x2, y2)
    depth_values = [] # Derinlik değerlerini saklamak için liste

    # Kamera çözünürlüğünü al
    cam_info = zed.get_camera_information()
    width = cam_info.camera_configuration.resolution.width
    height = cam_info.camera_configuration.resolution.height


    # Dikdörtgen çizilen kutu içerisindeki point cloud verilerini al ve derinlik değerini median al


    step = max(1, min((x2 - x1) // 10, (y2 - y1) // 10))  # Step : 

    for x in range(x1, x2, step):
        for y in range(y1, y2, step):
            if 0 <= y < height and 0 <= x < width:  
                depth_value = sl.Mat()

                #Her bi pixelin derinlik değerini al
                zed.retrieve_measure(depth_value, sl.MEASURE.DEPTH)
                depth_image = depth_value.get_data()
                depth = depth_image[y, x]
                
                #Eğer derinlik değeri geçerli ve 0 ile 1000 arasında ise ekle
                if (
                    depth > 0 and depth < 1000
                ):  # Filter out invalid and extreme depth values
                    depth_values.append(depth)

    # Eğer derinlik değerleri varsa
    if depth_values:

        
        # Tüm dikdörtgen içerisindeki derinlik değerlerini sırala ve ortanca değeri al.
        median_depth = sorted(depth_values)[len(depth_values) // 2]

        # 3B noktayı 2B piksel koordinatlarına dönüştürme 
        fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx

        # Genişlik ve yükseklik hesapla.
        width = abs(x2 - x1) * median_depth / fx
        height = abs(y2 - y1) * median_depth / fx
        return width, height, median_depth
    return None, None, None


def detect_objects(model, frame, conf_threshold=0.5):
    """
    YOLO modelini kullanarak nesneleri tespit eder ve sınırlayıcı kutuları döndürür.
    """

    # Eğer resim 4 kanallı ise (RGBA) 3 kanallıya dönüştür
    if frame.shape[2] == 4:  # If image has 4 channels (RGBA)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    # Modeli çalıştır ve sonuçları al
    results = model(frame, conf=conf_threshold)
    rects = []

    #! Results: [image_id, label, conf, x_min, y_min, x_max, y_max]

    for result in results:
        boxes = result.boxes

        # Her bir kutu için
        for box in boxes:
            # Eğer kutu'nun sınıfı 'kitap' veya 'kutu' ise
            
            if box.cls[0] in [73, 84]: # 73: book, 84: box

                #Koordinatları al
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = [int(x1), int(y1), int(x2), int(y2)]

                # Eğer kutu bir dikdörtgen ise listeye ekle
                if is_rectangular_box(bbox):
                    rects.append(bbox)

    return rects


def draw_rectangle_with_text(frame, bbox, text):
    """
    Tespit edilen dikdörtgen'lerin ekranda çizilmesi ve FPS değerinin eklenmesi
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Dikdörtgen çiz
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    # Arka planı yeşil olan metin kutusu çiz
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_w, y1), (0, 255, 0), -1)

    # Metni ekle
    cv2.putText(
        frame, text, (x1 + 1, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )
    cv2.putText(
        frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )


def main():
    global exit_signal, run_signal, i, detections

    ##########################################!KONFIGURASYON CAMERA ##################################################################
    # ZED kamera nesnesi oluştur ve konfigürasyonları ayarla
    zed = sl.Camera() # ZED kamerayı temsil eden nesneyi oluştur


    init_params = sl.InitParameters()  # Başlangıç parametrelerini oluştur

# DEPTH MODE : NEURAL= Yapay zeka destekli derinlik modu
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL

# COORDINATE UNITS : CM= Santimetre (DAHA HASSAS ÖLÇÜM İÇİN )
    init_params.coordinate_units = sl.UNIT.CENTIMETER

# CAMERA RESOLUTION : HD720= 1280x720 çözünürlük
    init_params.camera_resolution = (
        sl.RESOLUTION.HD720
    )  

#? Kamerayı başlat
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        logging.error(f"Camera Open : {repr(status)}. Exit program.")
        return
    ###########################!########################################################################################################
    
    
    try:
        # Modeli yükle
        model = load_model()
        logging.info("Model loaded successfully")

        # Kamera parametrelerini ayarla
        runtime_params = sl.RuntimeParameters()
        image = sl.Mat() # Kameradan alınan görüntüyü saklamak için Mat nesnesi


        logging.info("Object detection: Running...")

        # FPS hesaplama
        fps_start_time = time.time()
        fps = 0
        frame_count = 0


        # Q basılmadığı sürece
        while not exit_signal:

            # Kamera görüntüsünü al
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                

                # FPS Hesaplama
                frame_count += 1
                if frame_count >= 30:  # Update FPS every 30 frames
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    frame_count = 0

                # Kameradan alınan görüntüyü al
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()

                # Nesneleri tespit et
                rects = detect_objects(model, frame)
                detections = rects

                # Tespit edilen kutuların koordinatlarını ve ölçümlerini ekrana çiz
                for bbox in detections:
                    width, height, depth_val = measure_box(zed, bbox)
                    if width and height and depth_val:
                        text = (
                            f"W: {width:.1f}cm, H: {height:.1f}cm, D: {depth_val:.1f}cm"
                        )
                        draw_rectangle_with_text(frame, bbox, text)

                # Display FPS
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    4,
                )
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )



                # Görüntü başka bir renk uzayında ise (RGBA), RGB'ye dönüştür

                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                cv2.imshow("ZED | 2D View", display_frame)

                # Klavye işaretlerini al
                key = cv2.waitKey(1)

                # 's' tuşuna basıldığında ölçüm yap ve .txt dosyasına kaydet
                if key == ord("s"):  
                    if detections:
                        for bbox in detections:
                            i += 1
                            width, height, depth_val = measure_box(zed, bbox)
                            if width and height and depth_val:
                                with open("box_measurements.txt", "a") as f:
                                    f.write(
                                        f"Box{i} dimensions: Width={width:.2f}cm, Height={height:.2f}cm, Depth={depth_val:.2f}cm\n"
                                    )
                                logging.info(
                                    f"Box{i} measured and saved: {width:.2f}cm x {height:.2f}cm x {depth_val:.2f}cm"
                                )

                # 'q' veya 'ESC' tuşuna basıldığında programı kapat
                elif key in (27, ord("q")):  # ESC or 'q' to exit
                    exit_signal = True

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    
    
    # Kamerayı kapat ve programı sonlandır
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # YOLO modelinin güvenlik eşiği
    parser.add_argument("--conf", type=float, default=0.6, help="confidence threshold")
    opt = parser.parse_args()
    main()
