#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image
number=[]
# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "/media/jinesh/ACB6A202B6A1CCE0/Academics/8thSemester/CNN/python_files/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.createLBPHFaceRecognizer()
def train(trainpath):
    #trainpath = '/home/jinesh/Downloads/Varun Alia/cropped'
    #path = './yalefaces'
    # Call the get_images_and_labels function and get the face images and the
    # corresponding labels
    images, labels = get_images_and_labels(trainpath)
    cv2.destroyAllWindows()

    # Perform the tranining
    recognizer.train(images, np.array(labels))

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not (f.endswith('.test.jpg'))]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    print image_paths
    print "++++++++++++++++++++++++++++++++++++++++++++++++++"
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
	print image_path
        for (x, y, w, h) in faces:
            print x,y,w,h
            if(w<300 and h<300):
                pass
            else:
                print "face Detected"
                images.append(image[y: y + h, x: x + w])
                labels.append(nbr)
                cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
                cv2.waitKey(10)
    # return the images list and labels list
    #print images
    print "\n -----------------------------------------------"
    global number
    number = list(set(labels))
    return images, labels

# Path to the Yale Dataset



def test(image_path):
    #testpath = '/home/jinesh/Downloads/Varun Alia/cropped'
    #file1 = open(outputpath,"w")
    # Append the images with the extension .sad into image_paths
    #image_paths = [os.path.join(testpath, f) for f in os.listdir(testpath) if f.endswith('.test.jpg')]
    #for image_path in image_paths:
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
            print x,y,w,h
            if(w<300 and h<300):
                pass
            else:
                nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
                #nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
                #print number
                for nbr_actual in number:
                    if nbr_actual == nbr_predicted:
                        #file1.write("Actual :"+ str(nbr_actual)+" Predicted :"+str(nbr_predicted)+"\n" )
                        print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
                        return nbr_predicted
                    else:
                        print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
                    cv2.imshow("Reconizing Face", predict_image[y: y + h, x: x + w])
                    cv2.waitKey(5000)
    return -1
    #file1.close()
#train()
#test()
