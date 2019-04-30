from label_image import *
import os, sys, errno
import cv2 as cv
import numpy as np
import random as rng

# user input image
path_to_image = sys.argv[1]

# print("Segmenting the Image...")

# IMAGE PREPROCESSING:

# load INPUT_IMAGE and generate threshold segmentation THRESH
input_image = cv.imread(path_to_image, 1)
grayscale_input = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(grayscale_input,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# force THRESH to have a white background for uniformity
if thresh[thresh.shape[0] - 1][thresh.shape[1] - 1] == 0:
  thresh = cv.bitwise_not(thresh)

# dilate THRESH and update name to RAW_SEGMENTATION of shapes
kernel = np.ones((3,3), np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN, kernel, iterations = 1)
dilated_thresh = cv.dilate(opening, kernel,iterations=4)
raw_segmentation = cv.cvtColor(dilated_thresh, cv.COLOR_GRAY2RGB)

grayscale_input = cv.cvtColor(raw_segmentation, cv.COLOR_BGR2GRAY)
edged = cv.Canny(grayscale_input,30,200)

# find COUNTOURS in EDGED.copy() since contours alter the image
contours, hierarchy = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)

# draw COUNTOURS on RAW_SEGMENTATION
cv.drawContours(raw_segmentation, contours, -1, (255, 255, 255), 3)

# create bounding rectangles localizing each shape
contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)

for i, c in enumerate(contours):
    contours_poly[i] = cv.approxPolyDP(c, 3, True)
    boundRect[i] = cv.boundingRect(contours_poly[i])

# fill in CONTOURS in RAW_SEGMENTATION
for i in range(len(contours)):
        color_white = (255, 255, 255)
        color_black = (0, 0, 0)
        cv.drawContours(raw_segmentation, contours_poly, i, color_black)
        cv.fillPoly(raw_segmentation, pts = contours_poly, color=color_black)

# extract segmented shapes by their bounding rectangle 
extracted_shapes = []
for i, c in enumerate(contours):
  x, y, w, h = cv.boundingRect(c)
  roi = raw_segmentation [y:y+h, x:x+w]
  roi = np.pad(roi, ((20, 20), (20, 20), (0, 0)), 'constant', constant_values=(255))
  extracted_shapes.append(roi)

# make resource directory to hold the extracted shapes as jpgs
directory_name = "resources"
try:
    os.makedirs(directory_name)
except OSError as exc: 
    if exc.errno == errno.EEXIST and os.path.isdir(directory_name):
        pass

# write shapes to RESOURCE
num_segments = 0
for i, shape in enumerate(extracted_shapes):
  num_segments += 1
  cv.imwrite(os.path.join(directory_name, 'shape_' + str(i) + '.jpg'), shape)

# print("Classifying the Shapes....")

# CLASSIFICATION:

# dictionary to keep track of quantity of each class 
classes = {
  "circle" : 0,
  "square" : 0,
  "triangle" : 0,
}

# global variables
model_file = "retrained_graph.pb"
label_file = "retrained_labels.txt"
input_height = 224
input_width = 224
input_mean = 128
input_std = 128
input_layer = "input"
output_layer = "final_result"

def classify_shape(file_name):
  """
  Classifies abstract shape from FILE_NAME and updates CLASSES accordingly
  """
  graph = load_graph(model_file)
  t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:
    start = time.time()
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    end=time.time()
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)

  prediction = labels[top_k[0]]

  if prediction == "circle":
    classes["circle"] += 1
  elif prediction == "square":
    classes["square"] += 1
  elif prediction == "triangle":
    classes["triangle"] += 1
  elif prediction == "circle square":
    classes["circle"] += 1
    classes["square"] += 1
  elif prediction == "circle triangle":
    classes["circle"] += 1
    classes["triangle"] += 1
  elif prediction == "square triangle":  
    classes["square"] += 1
    classes["triangle"] += 1

  return prediction

# classify each image
for i in range(num_segments):
  file_name = "./resources/shape_" + str(i) + ".jpg"
  pred = classify_shape(file_name)
  print(pred)

print("++++-----OCCURENCES-----++++")
print("s: {0}, c: {1}, t:{2}".format(classes["square"], classes["circle"], classes["triangle"]))

import time 
time.sleep(5)

# empty the resource directory since its no longer needed
filelist = [f for f in os.listdir(directory_name) if f.endswith(".jpg") ]
for f in filelist:
    os.remove(os.path.join(directory_name, f))