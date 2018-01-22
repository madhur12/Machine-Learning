import cv2
import numpy as np
import os
import csv

def kmean():
    return 1

feature_list = []
exceptions = open("C:\\Users\\Madhur\\Desktop\\Python\\ML\\Project\\ExceptionList","w")

#for root, dirs, files in os.walk('C:\\Users\\Madhur\\Desktop\\Python\\ML\\Project\\101_ObjectCategories\\', topdown = True):
for dir_name in os.listdir('C:\\Users\\Madhur\\Desktop\\Python\\ML\\Project\\101_ObjectCategories\\'):

#    label = root[root.rfind("\\") +1:]
#   print "\n\n",label,"\n"

    if label.startswith('a'):
        for images in files:
            try:
                #print root +'\\'+ images
                img = cv2.imread(root +'\\'+ images)
                gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                sift = cv2.SIFT()
                kp = sift.detect(gray,None)

                #img=cv2.drawKeypoints(gray,kp)

                #cv2.imwrite('C:\\Users\\Madhur\\Desktop\\Python\\ML\\Project\\sift_keypoints.jpg',img)

                #img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                #cv2.imwrite('C:\\Users\\Madhur\\Desktop\\Python\\ML\\Project\\sift_keypoints_rich.jpg',img)

                kp, des = sift.detectAndCompute(gray,None)
                #print des.shape

                #Vector Quantization using Mean
                quantized = np.empty(128)
                for col in np.arange(128):
                    quantized[col] = des[:,col].sum()/len(des)
                
                #Writing features to a file in csv format
                row = quantized.tolist()
                row.append(label)
                
                #print row
                feature_list.extend([row])

            except TypeError:
                print "Exception in image : ", root +'\\'+ images, "::", Exception.__class__.__name__, "\n", quantized, "\nDone Exception"
                exceptions.write(root +'\\'+ images + "\t:: " + Exception.__class__.__name__)
    
print feature_list

features = open("C:\\Users\\Madhur\\Desktop\\Python\\ML\\Project\\features.csv","wb")
writer = csv.writer(features)
writer.writerows(feature_list)

features.close()
exceptions.close()
