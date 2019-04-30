Japjot Singh - Solution to Uizard Shape Recognition Challenge

To use my algorithm in python shell run:
>>> cool_algorithm.py <image path string>

*The input image must have format .jpg*

Output will be in the following format:
`
++++-----OCCURENCES-----++++
s: 2, c: 1, t:1
`
where s, c, t corresponds to the number of square, circles and triangles respectively


Included Files:

cool_algorithm.py - script to run the algorithm and provide output
label_image.py - module used to classify shapes within the input image
resources -  a ghost directory used in a subroutine of cool_algorithm.py to temporarily store segments
retrained_graph.pb - model file
retrained_labels.txt - different classes in the model

Strategy:

My approach was to preprocess the original image (using openCV) to remove noise, extract the abstract shapes and then use a CNN to classify the abstract shapes. I utilized Kaggle's Four Shapes training set for the basic shapes (square, circle, triangle). I had to create my own training data for overlapping shapes (square circle, square triangle, circle triangle). I utilized adobe photoshop to create ~70 varying images of each overlapping shape and then used Augmentor module to augment (by transformation, distortion, noise) my overlapping dataset to ~1000 images per class. I then utilized transfer learning on a pretrained mobilenet model to create my final model. 
