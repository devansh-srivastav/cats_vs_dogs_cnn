# Predicting Cat vs Dog

# Importing Libraries
import tensorflow as tf
import numpy as np

# Loading the model
classifier = tf.keras.models.load_model('catsVSdogs.h5')

# Loading and Processing Test Image
test_image = tf.keras.preprocessing.image.load_img('dataset/test_model/cat_or_dog.jpg', target_size = (64, 64))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

# Predicting Result
result = classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'