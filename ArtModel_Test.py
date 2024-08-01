import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import cv2

class LabelReader():
    def __init__(self):
        self.label = self.readlabel()

    def readlabel(self):
        print("Reading Label...")
        # file = open("data/artist.csv")
        file = open("data/artist.csv", "r", encoding="utf-8")
        labels = []
        file.readline()     # 첫번째 헤더는 건너뜀
        for line in file:
            splt = line.split(",")
            art = splt[1]

            labels.append(art)

        return labels



EPOCHS = 20
lb = LabelReader()

model = keras.models.load_model('./data/model/artist_a.h5')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = frame[0:600, 0:400]
    # test = cv2.cvtColor(cv2.imread())
    test = cv2.resize(frame,(224, 224))
    cv2.imshow('input', test)
    test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    test = test[np.newaxis, :, :, :]
    predict = model(test)
    predict_number = np.argmax(predict)
    print(str(predict))
    cv2.putText(frame, lb.label[predict_number], (100, 100), cv2.QT_FONT_NORMAL,1.4, (0, 0, 0), 2)
    cv2.imshow('cam', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

print(lb.label[predict_number])
