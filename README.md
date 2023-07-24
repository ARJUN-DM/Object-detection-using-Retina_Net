# Object-detection-using-Retina_Net

## Abstract:
The provided code is a Python script for performing real-time object detection on a video using a pre-trained RetinaNet model with a ResNet-50 backbone. The script uses the OpenCV library to read the input video, process each frame through the model, draw bounding boxes around detected objects, and display the resulting frame with object labels and inference time.

Abstract steps of the code:

1. Load the pre-trained RetinaNet model with the ResNet-50 backbone from the 'resnet50_coco_best_v2.1.0.h5' file.

2. Open the input video ('1410202020543050_20201014205430.mp4') using OpenCV and get its frames per second, width, and height.

3. Create an output video writer to save the processed video with bounding boxes.

4. Enter a continuous loop to read frames from the input video.

5. Calculate the inference time and frames per second for each frame.

6. Preprocess the frame for inference, including resizing and normalizing.

7. Make predictions on the frame using the loaded RetinaNet model.

8. Adjust the threshold for object detection and draw bounding boxes and labels for objects with a confidence score greater than 0.5.

9. Display the frame with bounding boxes and inference time.

10. Save the processed frame with bounding boxes to the output video.

11. Resize and display the frame with 'cv2.imshow'.

12. Exit the loop when 'q' key is pressed.

13. Release the video capture and video writer, and close all OpenCV windows.

The script aims to showcase real-time object detection using a pre-trained model and provides visual feedback on the detected objects, FPS, and inference time for each frame in the input video. The output video ('output_video.mp4') will contain the processed frames with bounding boxes around detected objects.

## System configurations
<OL>
  Processor : Intel® Core™ i5-6200U CPU @ 2.30GHz × 4
  Memory : 16GB
  Graphics : Mesa Intel® HD Graphics 520 SKL GT2
  Disk capacity : 480.1 GB
  OS Name : Ubuntu 20.04.6 LTS
</OL>
