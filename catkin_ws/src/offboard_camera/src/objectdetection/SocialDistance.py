# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
'''Find Social Distance violations using Tensorflow Lite and Yolov4 model

SocialDistance(model_path, label_path)

load_image(image_path)

annotated_image, number_of_violation = process_image([min_distance])
'''


# import the necessary packages
import cv2
from PIL import Image
import re

import numpy as np
from numpy import mean
from numpy import std
from scipy.spatial import distance as dist
import platform
import os.path

PLATFORM = platform.machine()
if PLATFORM in ['arm','arm64','aarch64'] :
    import tflite_runtime.interpreter as tflite
else :
    import tensorflow.lite as tflite

DEFAULT_MIN_DISTANCE = 50 # pixel, TODO: calc with elevation and camera focal

class SocialDistance:
    def __init__(self, model_path : str, label_path : str):
        self.model_path = model_path
        self.label_path = label_path

        self.interpreter = self.__load_model(model_path)
        self.labels = self.__load_labels(label_path)
        
        input_details = self.interpreter.get_input_details()
        # Get Width and Height
        # _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
        input_shape = input_details[0]['shape']
        self.image_height = input_shape[1]
        self.image_width = input_shape[2]

        # Get input index
        self.input_index = input_details[0]['index']
        self.min_distance = DEFAULT_MIN_DISTANCE
        self.original_image = None


    def __load_labels(self,label_path):
        """Returns a list of labels"""
        with open(label_path) as f:
            labels = {}
            for line in f.readlines():
                m = re.match(r"(\d+)\s+(\w+)", line.strip())
                labels[int(m.group(1))] = m.group(2)
            return labels


    def __load_model(self, model_path):
        """Load TFLite model, returns a Interpreter instance."""
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    # load image from disk
    def load_image(self, image_path : str ):
        """loads the image and convert into yolov4 
        Args:
            image_path: str representing image path

        Returns:
            None
        Raises:
            ValueError: Unsupported value.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError()

        img = cv2.imread(image_path)

        self.original_image = img
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        #new_image = cv2.convertScaleAbs(original_image, alpha=3.5, beta=60)
        image_data = cv2.resize(self.original_image, (self.image_width, self.image_height))
    
        self.image_data = image_data / 255.
              
        return img

    # set image from camera capture
    def use_image(self, frame) :
        self.original_image = frame
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        #new_image = cv2.convertScaleAbs(original_image, alpha=3.5, beta=60)
        image_data = cv2.resize(self.original_image, (self.image_width, self.image_height))
    
        self.image_data = image_data / 255.

    def process_image(self, min_distance : int = None):
        """Process an image, Return a list of detected class ids and positions
            Args:
                min_distance: int representing the distance between people. default 50 px
            Returns:
                annotated_image: annotaed image with boxes if any person is detected
                number_of_violation: number of social distance violation
            Raises:
                None.
        """
        # Process
        images_data = []
        for i in range(1):
            images_data.append(self.image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        self.interpreter.set_tensor(self.input_index, images_data)
        self.interpreter.invoke()

        self.min_distance = self.min_distance if  min_distance == None else min_distance
        # Get outputs
        output_details = self.interpreter.get_output_details()
        predictions = [self.interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        top_result, number_of_violation = self.__handle_predictions(predictions[0], predictions[1], confidence=0.3, iou_threshold=0.2)

        # annotate image with boxes
        annotated_image = self.__annotate_objects(top_result) if not (top_result is None) else self.original_image
        return annotated_image, number_of_violation

    def __annotate_objects(self, results):
        """Draws the bounding box and label for each object in the results."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255,255, 2555) 
        
        orig_img_h, orig_img_w, _ = self.original_image.shape
        pred_img_h, pred_img_w, _ = self.image_data.shape
        ratio_h, ratio_w = orig_img_h/pred_img_h , orig_img_w/pred_img_w

        font_size = 0.2 * ratio_h
        font_thickness = int(0.1 * ratio_h)
        box_thickness = int(0.8 * ratio_h)
        max_area = 0
        annotated_image = self.original_image.copy()
        for obj in results:
            # Convert the bounding box figures from relative coordinates
            # to absolute coordinates based on the original resolution
            
            xmin, ymin, xmax, ymax = obj['bounding_box']
            area = (xmax - xmin) * (ymax - ymin)
            if ( area > max_area) :
                max_area = area
                coor = obj['bounding_box']
            xmin = int(xmin * ratio_w)
            xmax = int(xmax * ratio_w)
            ymin = int(ymin * ratio_h)
            ymax = int(ymax * ratio_h)

            cv2.putText(annotated_image, self.labels[obj['class_id']], (xmin, ymin-5), font, font_size, color, font_thickness)
            cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), obj['color'], box_thickness)
        return annotated_image


    def __removeOutliers(self, results):
        boxAreas = []
        for obj in results:
            xmin, ymin, xmax, ymax = obj['bounding_box']
            area = (xmax - xmin) * (ymax - ymin)
            boxAreas.append(area)
        area_mean, area_std = mean(boxAreas), std(boxAreas)
        cut_off = area_std *3
        lower, upper = area_mean - cut_off, area_mean + cut_off
        results_outliers_removed = [results[i] for i, x in enumerate(boxAreas)  if x >= lower and x <= upper]

        return results_outliers_removed

    def __distance_violation(self, results):
        violate = set()
        if len(results) >= 2:
            centroids = np.array([obj['centroid'] for obj in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")  # compute Euclidean distances
            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < DEFAULT_MIN_DISTANCE:
                        violate.add(i)
                        violate.add(j)
                        results[i]['color'] = (0,0,255)
                        results[j]['color'] = (0,0,255)
        return results, len(violate)

    def __handle_predictions(self, boxes, scores, confidence=0.3, iou_threshold=0.5):
        box_classes = np.argmax(scores, axis=-1)
        box_class_scores = np.max(scores, axis=-1)
        pos = np.where(box_class_scores >= confidence)
    
        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        n_boxes, n_classes, n_scores = self.__nms_boxes(boxes, classes, scores, iou_threshold)

        if n_boxes:
            boxes = np.concatenate(n_boxes)
            classes = np.concatenate(n_classes)
            scores = np.concatenate(n_scores)
            count = classes.size

            results = []
            for i in range(count):
                if classes[i] == 0 : #and scores[i] >= confidence:
                    center_x, center_y, self.image_width, self.image_height = boxes[i]
                    w2 = self.image_width / 2
                    h2 = self.image_height / 2
                    x0 = int(center_x - w2)
                    y0 = int(center_y - h2)
                    x1 = int(center_x + w2)
                    y1 = int(center_y + h2)

                    result = {
                        'centroid' : boxes[i],
                        'bounding_box': [x0, y0, x1, y1],
                        'class_id': classes[i],
                        'score': scores[i],
                        'color' : (255,0, 0)
                    }
                    results.append(result)
            results = self.__distance_violation(self.__removeOutliers(results))
            return results

        else:
            return None, 0

    def __nms_boxes(self, boxes, classes, scores, iou_threshold):
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            x = b[:, 0]
            y = b[:, 1]
            w = b[:, 2]
            h = b[:, 3]

            areas = w * h
            order = s.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x[i], x[order[1:]])
                yy1 = np.maximum(y[i], y[order[1:]])
                xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
                yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

                w1 = np.maximum(0.0, xx2 - xx1 + 1)
                h1 = np.maximum(0.0, yy2 - yy1 + 1)

                inter = w1 * h1
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                inds = np.where(ovr <= iou_threshold)[0]
                order = order[inds + 1]

            keep = np.array(keep)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
        return nboxes, nclasses, nscores

if __name__ == "__main__":
    model_path = 'data/yolov4-416.tflite'
    label_path = 'data/coco_labels.txt'
    image_path = 'data/staugustine.jpeg'
    sd = SocialDistance(model_path, label_path)
    print('SocialDistance is initialized' )