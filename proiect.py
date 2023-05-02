import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import BatchNormalization, Dropout, MaxPooling2D, Dense, Conv2D, Flatten, AveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model


def loadData(folder):
    X = []
    y = []

    for label, subfolder in enumerate(['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']):
        for imgFile in tqdm(os.listdir(os.path.join(folder, subfolder))):
            img_path = os.path.join(os.path.join(folder, subfolder), imgFile)
            image = cv2.imread(img_path)
            image = cv2.resize(image, [120, 120])

            X.append(image / 255.0)
            y.append(label)

    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')
    return X, y


X_train, y_train = loadData('assets/dataset2-master/dataset2-master/images/TRAIN')
X_test, y_test = loadData('assets/dataset2-master/dataset2-master/images/TEST')
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.7)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)
X_train, y_train = shuffle(X_train, y_train, random_state=616)
classNames = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

_, train_counts = np.unique(np.argmax(y_train, axis=1), return_counts=True)
_, val_counts = np.unique(np.argmax(y_val, axis=1), return_counts=True)
_, test_counts = np.unique(np.argmax(y_test, axis=1), return_counts=True)

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(120, 120, 3)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))


model.add(Dense(units=4, activation='softmax'))


model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])



datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)
checkpoint = ModelCheckpoint(
    filepath='model-{epoch:02d}-{val_accuracy:.2f}.h5',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
earlystop = EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True)
lrReduction = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=2,
    verbose=1,
    factor=0.3,
    min_lr=0.00001)
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=20, validation_data=[X_val, y_val],
                    callbacks=[lrReduction, checkpoint])
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
axs[0].plot(epochs, acc, 'b-o', label='Training acc')
axs[0].plot(epochs, val_acc, 'r-o', label='Validation acc')
axs[0].set_title('Training and validation accuracy')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Accuracy')
axs[0].legend()

axs[1].plot(epochs, loss, 'b-o', label='Training loss')
axs[1].plot(epochs, val_loss, 'r-o', label='Validation loss')
axs[1].set_title('Training and validation loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].legend()

plt.show()
y_pred = model.predict(X_test)
test_result = model.evaluate(X_test, y_test)

print("Pierderea in acest model este : ", test_result[0])
print("Precizia modelului este: ", test_result[1] * 100, "%")
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test_labels, y_pred_labels)

classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

plt.figure(figsize=(10, 10))
sns.heatmap(
    cm,
    cmap='YlGnBu',
    linecolor='black',
    linewidth=1,
    annot=True,
    fmt='',
    xticklabels=classes,
    yticklabels=classes)
model.save('assets/working/final.h5')
