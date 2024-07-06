# Cat vs Dog Classification Model

This project involves creating a convolutional neural network (CNN) to classify images of cats and dogs. The dataset used for training and validation is the "Dogs vs. Cats" dataset available on Kaggle.

## Steps to Execute

### 1. Set Up Kaggle API
1. Create a Kaggle API token and save it as `kaggle.json`.
2. Copy the token to the appropriate directory:
   ```python
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   ```

### 2. Download the Dataset
Download the "Dogs vs. Cats" dataset from Kaggle:
```python
!kaggle datasets download -d salader/dogs-vs-cats
```
Extract the dataset:
```python
import zipfile
zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()
```

### 3. Import Libraries
```python
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
```

### 4. Prepare the Data
Load and preprocess the training and validation datasets:
```python
train_ds = keras.utils.image_dataset_from_directory(
    directory='/content/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory='/content/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)

def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)
```

### 5. Build the CNN Model
Create a CNN with three convolutional layers:
```python
model = Sequential()

# First Layer
model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# Second Layer
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# Third Layer
model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.summary()
```

### 6. Compile and Train the Model
Compile the model and train it:
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
backup = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```

### 7. Plot Training and Validation Metrics
Visualize training and validation accuracy and loss:
```python
import matplotlib.pyplot as plt

plt.plot(backup.history['accuracy'], color='red', label='train')
plt.plot(backup.history['val_accuracy'], color='blue', label='validation')
plt.legend()
plt.show()

plt.plot(backup.history['loss'], color='red', label='train')
plt.plot(backup.history['val_loss'], color='blue', label='validation')
plt.legend()
plt.show()
```

### 8. Test the Model
Test the model with sample images:
```python
import cv2

# Test with a cat image
test_img = cv2.imread('/content/cat.jpg')
plt.imshow(test_img)
test_img = cv2.resize(test_img, (256, 256))
test_input = test_img.reshape((1, 256, 256, 3))
model.predict(test_input) # Output close to 0 indicates a cat

# Test with a dog image
test_img1 = cv2.imread('/content/dog.jpg')
plt.imshow(test_img1)
test_img1 = cv2.resize(test_img1, (256, 256))
test_input1 = test_img1.reshape((1, 256, 256, 3))
model.predict(test_input1) # Output close to 1 indicates a dog
```

## Conclusion
The CNN model successfully classifies images of cats and dogs. Batch normalization and dropout techniques are applied to improve accuracy and reduce overfitting. The model's performance is visualized through accuracy and loss plots.
