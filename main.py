from numpy.testing._private.utils import print_assert_equal
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
import cv2
import numpy as np
import Ui_gui
inputImagePath=''
def fileDialog():
    imagePath=QFileDialog.getOpenFileName()
    imagePath=imagePath[0]
    inputImagePath=imagePath
    inputImagePath=cv2.imread(inputImagePath)
    inputImagePath=cv2.resize(inputImagePath,(488,378))
    cv2.imwrite(f'Testing/{imagePath.split("/")[-1]}',inputImagePath)
    ui.inputImage.setPixmap(QtGui.QPixmap(imagePath))
    ui.outputImage.setPixmap(QtGui.QPixmap(imagePath))
    cv2.imwrite('input.jpg',inputImagePath)
def code():
    model= model_from_json(open('ModelArchitecture.json','r').read())
    model.load_weights('FacialModelWeights.h5')
    test_image=cv2.imread('input.jpg')
    gray_image=cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    faces=face_cascade.detectMultiScale(gray_image,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(test_image,(x,y),(x+w,y+h),(0,0,0))
        roi_gray=gray_image[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'sad')
        emotion_prediction = emotion_detection[max_index]
        print(emotion_prediction)
        font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        org=(70,70)
        color=(0,0,255)
        thickness=2
        image=cv2.putText(test_image,emotion_prediction,org,font,1,color,thickness,cv2.LINE_AA)
        cv2.imwrite('output.jpg',image)
        ui.outputImage.setPixmap(QtGui.QPixmap('output.jpg'))
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_gui.Ui_MainWindow()
    MainWindow.setFixedSize(1024, 490)
    ui.setupUi(MainWindow)
    app_icon = QtGui.QIcon()
    app_icon.addFile('icon.jpg', QtCore.QSize(16,16))
    MainWindow.setWindowIcon(app_icon)
    ui.browseButton.clicked.connect(fileDialog)
    ui.outputButton.clicked.connect(code)
    MainWindow.show()
    sys.exit(app.exec_())
 