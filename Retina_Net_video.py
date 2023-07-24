import cv2
import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
import time


model_path = 'resnet50_coco_best_v2.1.0.h5'
model = models.load_model(model_path, backbone_name='resnet50')

video_path = '1410202020543050_20201014205430.mp4'
video = cv2.VideoCapture(video_path)
fps1 = video.get(cv2.CAP_PROP_FPS)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video writer
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, fourcc, fps1, (frame_width, frame_height))

fps = 0
start_time = 0

while True:
    # Read the next frame from the video
    ret, frame = video.read()
    end_time = time.time()
    inf_time = end_time - start_time
    inf_time1 = inf_time*1000
    fps = 1 / inf_time
    start_time = end_time
    fps_text = "FPS: {:.2f}".format(fps)
    inf_text = "Inference_time: {:.2f}".format(inf_time1)

    cv2.putText(frame, fps_text, (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(frame, inf_text, (10, 150),cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)

    if not ret:
        break

    # Preprocess the frame for inference
    image = preprocess_image(frame)
    image, scale = resize_image(image)

    # Make predictions on the frame
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # Apply the scale factor to the bounding boxes
    boxes /= scale

    # Display the predicted objects
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.5:  # Adjust the threshold for object detection
            break
        box = box.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}: {score:.2f}', (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    output_video.write(frame)
    frame1 = cv2.resize(frame, (1280, 720))
    cv2.imshow('Object Detection', frame1)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video.release()
output_video.release()
cv2.destroyAllWindows()
