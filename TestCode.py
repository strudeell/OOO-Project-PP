import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
#
# def emnist_predict_img(model_path, image_array):
#     model = keras.models.load_model(model_path)
#     predictions = model.predict(image_array)
#     predicted_label = np.argmax(predictions)
#     return predicted_label
#
# #image_path = "letters/3.1.png"
# image_path = "/content/Щ.3.jpg"
#
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
# resized_image = cv2.resize(image, (28, 28))
#
# inverted_image = cv2.bitwise_not(resized_image)
#
# normalized_image = inverted_image / 255.0
#
# image_array = np.expand_dims(normalized_image, axis=0).reshape(1, 28, 28, 1)
#
# predicted_label = emnist_predict_img("neiro2(3).h5", image_array)
# print("Predicted label:", predicted_label)
#
# # blank_path = tf.keras.utils.get_file('blank', origin=blank)
# blank = "/content/Щ.3.jpg"
# img = tf.keras.utils.load_img(
#     blank, target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)
#
# # make predictions
# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
# print(predictions)
#
# # print inference result
# print("На изображении скорее всего {} ({:.2f}% вероятность)".format(
#   class_names[np.argmax(score)],
#   100 * np.max(score)))
#
# # show the image itself
# img.show()
class_names =  ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']
blank = "letters/3.1.png"
def load_model():
    model = tf.keras.models.load_model('neiro2(3).h5', compile=False)
    return model
#model_path = load_model() #"neiro2(3).h5"
# blank_path = tf.keras.utils.get_file('blank', origin=blank)
model = load_model()#keras.models.load_model(model_path)
img = tf.keras.utils.load_img(
    blank, target_size=(28, 28)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# make predictions
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(predictions)

# print inference result
print("На изображении скорее всего {} ({:.2f}% вероятность)".format(
  class_names[np.argmax(score)],
  100 * np.max(score)))

# show the image itself
img.show()