from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops

mypath='not'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
f1=0
f2=0
f3=0
f4=0
l=[0.0,0.0,0.0]
count=86
flag=6
label=0
for n in range(0, len(onlyfiles)):
  count=count+1
  if(flag!=9):
      flag=flag+1
  else:
      flag=0
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) )

 #Feature 1: Amount of green colour in the picture.
  hsv = cv2.cvtColor(images[n], cv2.COLOR_BGR2HSV)
  boundaries = [([30,0,0],[70,255,255])]
  for (lower, upper) in boundaries:
      mask = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))
      ratio_green = cv2.countNonZero(mask)/(images[n].size/3)
      f1=np.round(ratio_green, 2)
      print(f1)
 #Feature 2: Amount of non-green clour in the picture
  f2=1-f1
  print(f2)

 #Feature 3: Periphery length
  hsv = cv2.cvtColor(images[n], cv2.COLOR_BGR2HSV)
  hsv = cv2.split(hsv)
  gray = hsv[0]

  gray = cv2.GaussianBlur(gray, (3,3), sigmaX=-1, sigmaY=-1)

  ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

  contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

  cv2.drawContours(images[n], contours, -1, (255,0,0), thickness = 2)
  perimeter=0
  for c in contours:
       perimeter = perimeter+cv2.arcLength(c,True)
  f3=perimeter/15000
  print(f3)
 #Feature 4: Contrast
  img=cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY)
  g=greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
#print (g)

  contrast = greycoprops(g, 'contrast')
  f4=contrast[0][0]+contrast[0][1]+contrast[0][2]+contrast[0][3]
  f4=f4/2000000000
  print(f4)

  l=[count,flag,label,f1,f2,f3,f4]
 
  out = open('out.csv', 'a')
  for row in l:
      out.write('%f;' % row)
  out.write('\n')
  out.close()
print(images.size)

