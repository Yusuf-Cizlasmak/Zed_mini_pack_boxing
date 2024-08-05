import cv2
import pyzed.sl as sl
import numpy as np
import math

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create and configure InitParameters object
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units

    # Open the camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        exit()

    # Create RuntimeParameters object
    runtime_parameters = sl.RuntimeParameters()

    # Create Mat objects to store images and depth
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

    while True:
        # Grab an image and its corresponding depth map
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve depth map
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # Get the image data
            image_ocv = image.get_data()

            # Get the image center coordinates
            height, width = image_ocv.shape[:2]
            x = width // 2
            y = height // 2

            # Get the point cloud value at the center of the image
            err, point_cloud_value = point_cloud.get_value(x, y)

            # Compute the Euclidean distance from the camera to the point
            if math.isfinite(point_cloud_value[2]):
                distance = math.sqrt(point_cloud_value[0]**2 +
                                     point_cloud_value[1]**2 +
                                     point_cloud_value[2]**2)
                distance_text = f"Distance: {distance:.2f} mm"
            else:
                distance_text = "Distance: Not Available"

            # Draw the distance text on the image
            cv2.putText(image_ocv, distance_text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Normalize depth map for display
            depth_ocv = depth.get_data()
            depth_ocv = np.clip(depth_ocv, 0, 2000)  # Clip values for better visualization
            depth_ocv = depth_ocv / 2000.0 * 255
            depth_ocv = depth_ocv.astype(np.uint8)

            # Display the images
            cv2.imshow("ZED Mini Camera - RGB Image", image_ocv)
            cv2.imshow("ZED Mini Camera - Depth Map", depth_ocv)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Close the camera
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
