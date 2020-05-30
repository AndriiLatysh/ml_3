import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras.models as keras_models


np.set_printoptions(linewidth=200)

while True:

    image_path = input("Enter path to the image: ")

    digit_image = cv2.bitwise_not(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY))

    print(digit_image)
    plt.imshow(digit_image, cmap="Greys")

    digit_recognition_model = keras_models.load_model("models/MNIST_CNN.h5")
    digit_image_prepared = digit_image.reshape(1, 28, 28, 1).astype("float32") / 255

    digit_prediction = digit_recognition_model.predict(digit_image_prepared)

    for z in range(len(digit_prediction[0])):
        print("{} -> {:.2f}%".format(z, digit_prediction[0][z] * 100))

    plt.show()


