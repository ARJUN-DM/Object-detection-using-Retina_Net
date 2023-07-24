# Object-detection-using-Retina_Net

## Abstract:
The provided code is a Python script for performing real-time object detection on a video using a pre-trained RetinaNet model with a ResNet-50 backbone. The script uses the OpenCV library to read the input video, process each frame through the model, draw bounding boxes around detected objects, and display the resulting frame with object labels and inference time.

Abstract steps of the code:
Load the pre-trained RetinaNet model with the ResNet-50 backbone from the 'resnet50_coco_best_v2.1.0.h5' file.Open the input video ('Your Input video') using OpenCV and get its frames per second, width, and height. You have to specify your input video file path name inside the video_path declaration.Create an output video writer to save the processed video with bounding boxes. Enter a continuous loop to read frames from the input video. Calculate the inference time and frames per second for each frame. Preprocess the frame for inference, including resizing and normalizing. Make predictions on the frame using the loaded RetinaNet model. Adjust the threshold for object detection and draw bounding boxes and labels for objects with a confidence score greater than 0.5. Display the frame with bounding boxes and inference time. Save the processed frame with bounding boxes to the output video. Resize and display the frame with 'cv2.imshow'. Exit the loop when 'q' key is pressed. Release the video capture and video writer, and close all OpenCV windows. The script aims to showcase real-time object detection using a pre-trained model and provides visual feedback on the detected objects, FPS, and inference time for each frame in the input video. The output video ('output_video.mp4') will contain the processed frames with bounding boxes around detected objects.

## System configurations

  >Processor : Intel® Core™ i5-6200U CPU @ 2.30GHz × 4
  
  >Memory : 16GB
  
  >Graphics : Mesa Intel® HD Graphics 520 SKL GT2
  
  >Disk capacity : 480.1 GB
  
  >OS Name : Ubuntu 20.04.6 LTS


## Packages and libraries required
> 1. Opencv-pythoon
> 2. Numpy
> 3. Keras-resnet
> 4. Keras-retinanet
> 5. Tensorflow

## Inference time and FPS examined in Retina_Net
---------------------------------------------
| Sl.no | Resolution | Inference Time | FPS |
|-------|------------|----------------|-----|
| 1.    | 640x480    | 4067.00        | 0.24|
| 2.    | 1280x720   | 4492.64        | 0.22|
| 3.    | 1920x1080  | 4916.05        | 0.20|
| 4.    | 2560x1440  | 4417.02        | 0.23|
| 5.    | 2048x1080  | 4192.61        | 0.24|
| 6.    | 3840x2160  | 4362.50        | 0.23|
| 7.    | 7680x4320  | 4797.06        | 0.21|
