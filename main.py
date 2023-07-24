import cv2
import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image


# Load the pre-trained RetinaNet model
model_path = 'resnet50_coco_best_v2.1.0.h5'
model = models.load_model(model_path, backbone_name='resnet50')

image_path = 'img2.jpeg'
image = cv2.imread(image_path)
image = preprocess_image(image)
image, scale = resize_image(image)

# Make predictions on the image
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

# Apply the scale factor to the bounding boxes
boxes /= scale

# Display the predicted objects
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    if score < 0.5:  # Adjust the threshold for object detection
        break
    box = box.astype(int)
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.putText(image, f'{label}: {score:.2f}', (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the resulting image
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
