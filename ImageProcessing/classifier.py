import numpy as np
from sklearn import svm
import csv
import argparse
import cv2
from skimage.feature import greycomatrix, greycoprops
 
parser =argparse.ArgumentParser()
parser.add_argument('input_img', help = 'the input image file')
args = parser.parse_args()
image = cv2.imread(args.input_img)
f1=0
f2=0
f3=0
f4=0

#Feature 1: Amount of green colour in the picture.
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
boundaries = [([30,0,0],[70,255,255])]
for (lower, upper) in boundaries:
      mask = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))
      ratio_green = cv2.countNonZero(mask)/(image.size/3)
      f1=np.round(ratio_green, 2)

#Feature 2: Amount of non-green clour in the picture
f2=1-f1

#Feature 3: Periphery length
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv = cv2.split(hsv)
gray = hsv[0]
gray = cv2.GaussianBlur(gray, (3,3), sigmaX=-1, sigmaY=-1)
ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
cv2.drawContours(image, contours, -1, (255,0,0), thickness = 2)
perimeter=0
for c in contours:
     perimeter = perimeter+cv2.arcLength(c,True)
f3=perimeter/15000

#Feature 4: Contrast
img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
g=greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
contrast = greycoprops(g, 'contrast')
f4=contrast[0][0]+contrast[0][1]+contrast[0][2]+contrast[0][3]
f4=f4/2000000000

filename = "out.csv"
 
# initializing the titles and rows list
fields = []
rows = []
 
# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
     
    # extracting field names through first row
    fields = next(csvreader)
 
    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)
 
    # get total number of rows
    print("Total no. of rows: %d"%(csvreader.line_num))
 
# printing the field names
print('Field names are:' + ', '.join(field for field in fields))
A=[]
Y=[]
#  printing first 5 rows
print('\nFirst 5 rows are:\n')
included =[3,4,5,6]
y1 = [2]
for row in rows:
        content = list(row[i] for i in included)
        A.append(content)
for row in rows:
        content = list(row[i] for i in y1)
        Y.append(content)
"""print(len(Y))
for i in range(183):
    print(Y[i])
"""
b = np.reshape(Y, (183))
print(b.shape)
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(A,b)
print(clf.predict([[f1,f2,f3,f4]]))
print(f1)
print(f2)
print(f3)
print(f4)

"""w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(A[:, 0], A[:, 1], c = b)
plt.legend()
plt.show()"""
