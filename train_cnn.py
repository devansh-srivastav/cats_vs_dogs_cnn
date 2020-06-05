# Dogs vs Cats Classifier using CNN

# Importing Libraries
import tensorflow as tf
import matplotlib.pyplot as plt

# Loading and Preprocessing Training Data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

# Loading and Preprocessing Test Data
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# Initialising CNN
classifier = tf.keras.models.Sequential()

# Adding Convolutional and Pooling Layers
classifier.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3),
                                      input_shape = (64,64,3), activation = 'relu'))
classifier.add(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2))
classifier.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'))
classifier.add(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2))
classifier.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'))
classifier.add(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2))

# Adding Flatenning Layer
classifier.add(tf.keras.layers.Flatten())

# Adding Fully Connected Layers
classifier.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
classifier.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

# Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting CNN
epoch_history = classifier.fit(x = training_set, validation_data = test_set,
                               epochs = 25, steps_per_epoch=8000//32, validation_steps=2000//32, workers = 16)

# Visualising Model
plt.plot(epoch_history.history['loss'])
plt.title('Training Loss Progress')
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.plot(epoch_history.history['accuracy'])
plt.title('Training Accuracy Progress')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.plot(epoch_history.history['val_loss'])
plt.title('Test Loss Progress')
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.plot(epoch_history.history['val_accuracy'])
plt.title('Test Accuracy Progress')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

# Saving Model
classifier.save('catsVSdogs.h5')
