import cv2
import torch
import pyzed.sl as sl
import numpy as np
import logging

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Buraya eğitilen model gelecek!


# ZED SDK initiliaze edilmesi
init_params = sl.InitParameters(depth_mode=sl.DEPTH_MODE.PERFORMANCE)



# Kameranın initiliaze edilmesi
zed = sl.Camera()
zed.open(init_params)



# Kameranın RGB ve Depth görüntülerini almak için kullanılacak olan runtime parametreler
runtime_parameters = sl.RuntimeParameters()



# image, depth ve point cloud matrislerini oluştur
image = sl.Mat()
depth = sl.Mat()
point_cloud = sl.Mat()
logging.info("{}, {}, {}".format(image, depth, point_cloud))

while True:
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        # Convert ZED image to OpenCV format
        img = image.get_data()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Perform object detection
        results = model(img)

        # Process detection results
        for detection in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get depth data
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            err, point = point_cloud.get_value(x_center, y_center)

            distance = np.sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2])
            distance_str = f"{distance:.2f} m"
            # Calculate width and height of the box
            x1_err, x1_point = point_cloud.get_value(x1, y_center)
            x2_err, x2_point = point_cloud.get_value(x2, y_center)
            y1_err, y1_point = point_cloud.get_value(x_center, y1)
            y2_err, y2_point = point_cloud.get_value(x_center, y2)

            if x1_err == sl.ERROR_CODE.SUCCESS and x2_err == sl.ERROR_CODE.SUCCESS and y1_err == sl.ERROR_CODE.SUCCESS and y2_err == sl.ERROR_CODE.SUCCESS:
                width = np.sqrt((x2_point[0] - x1_point[0]) ** 2 + (x2_point[1] - x1_point[1]) ** 2 + (x2_point[2] - x1_point[2]) ** 2)
                height = np.sqrt((y2_point[1] - y1_point[1]) ** 2 + (y2_point[2] - y1_point[2]) ** 2)
                width_str = f"Width: {width:.2f} m"
                height_str = f"Height: {height:.2f} m"
            else:
                width_str = "Width: N/A"
                height_str = "Height: N/A"

            # Draw bounding box and depth information
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, distance_str, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, width_str, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, height_str, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the result
        cv2.imshow("ZED + YOLO", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

zed.close()