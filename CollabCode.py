import keras
import tensorflow as tf
import numpy as np

class_names = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']
model_path = "model_epoch_50(1).h5"
img_path = 'ru1.jpg'
#model = keras.models.load_model(model_path)
def load_model():
    model = tf.keras.models.load_model(model_path, compile=False)
    return model
img = tf.keras.utils.load_img(
    img_path, target_size=(278, 278)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# make predictions
predictions = load_model().predict(img_array)
score = tf.nn.softmax(predictions[0])
print(predictions)

# print inference result
print("На изображении скорее всего {} ({:.2f}% вероятность)".format(
  class_names[np.argmax(score)],
  100 * np.max(score)))

# show the image itself
img.show()