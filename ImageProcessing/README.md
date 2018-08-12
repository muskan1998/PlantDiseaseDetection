## Plant disease detction using image processing in OpenCV and SVM Classifier

The proposed framework can be used to identify diseased leaves.
Automatic detection of diseasesed plants is an important research topic since it is able to automatically detect the diseasesed plants from the symptoms that appear on the plant leaves.

## Method:
The following features of the image are calculated using OpenCV in features.py creating train.csv and test.csv from taining and test data respectively -

Feature 1: Colour ratio of green colour is calculated using mask of green threshold on HSV conversion of given RGB image.

Feature 2: The ratio of non-green part is calculated by subtracting the green ratio from total.

Feature 3: Using Mean Shift and Edge Detection using contour functionc, perifery is calculated.

Feature 4: Image contrast is calculated using Grey Level Co-occurence Matrix.
  

Using classifier.py the SVM classifier is trained on train.csv and then tested on test.csv and the accuracy of both is printed.
  
## Result:
  As a result 73.3% training accuracy and 67.2% test accuracy is achieved on 500 training examples and 85 test examples.
