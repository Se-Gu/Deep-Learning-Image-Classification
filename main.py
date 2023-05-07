import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.metrics import Precision, Recall
import copy


# Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# Split dataset into training, validation, and test sets
val_size = 0.2
val_samples = int(len(x_train) * val_size)

x_val = x_train[:val_samples]
y_val = y_train[:val_samples]
x_train = x_train[val_samples:]
y_train = y_train[val_samples:]

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
num_classes = 100
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

# Data augmentation
datagen_train = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

datagen_train.fit(x_train)

# Print dataset shapes
print('Train data shape:', x_train.shape, y_train.shape)
print('Validation data shape:', x_val.shape, y_val.shape)
print('Test data shape:', x_test.shape, y_test.shape)

# Define the model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Print the model summary
model.summary()

# Compile the model with a categorical cross-entropy loss function and Adam optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

# Set the batch size and number of epochs
batch_size = 64
epochs = 1

# Train the model on the training set
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    verbose=1)

scores = model.evaluate(x_val, y_val, verbose=1)

# Evaluate the model on the validation set
y_pred = model.predict(x_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')

# Print the evaluation results
print('Validation loss:', scores[0])
print('Validation accuracy:', scores[1])
print('Validation precision:', precision)
print('Validation recall:', recall)


# Define the ablation study configurations
configurations = [
    {'name': 'no_dropouts', 'layers': ['DROPOUT', 'DROPOUT_1', 'DROPOUT_2']},
    {'name': 'no_second_dropout', 'layers': ['DROPOUT_1']},
    {'name': 'half_filters', 'filters_factor': 0.5, 'layers': []},
    {'name': 'double_filters', 'filters_factor': 2.0, 'layers': []},
    {'name': 'half_dense_units', 'dense_units_factor': 0.5, 'layers': []},
    {'name': 'double_dense_units', 'dense_units_factor': 2.0, 'layers': [] },
]

# Perform ablation study
for config in configurations:
    print(f"Ablation study: {config['name']}")

    # Create a copy of the original model
    modified_model = Sequential.from_config(model.get_config())

    # Iterate over the layers of the model and modify them according to the current configuration
    for layer_index, layer in enumerate(modified_model.layers):
        if layer.name in config['layers']:
            if isinstance(layer, Dropout):
                modified_model.layers[layer_index] = Dense(layer.units, activation=layer.activation)
            else:
                modified_model.layers[layer_index] = None
        elif isinstance(layer, Conv2D):
            modified_model.layers[layer_index] = Conv2D(int(layer.filters * config.get('filters_factor', 1.0)),
                                                        layer.kernel_size,
                                                        padding=layer.padding,
                                                        activation=layer.activation)
        elif isinstance(layer, Dense):
            modified_model.layers[layer_index] = Dense(int(layer.units * config.get('dense_units_factor', 1.0)),
                                                       activation=layer.activation)

    # Remove None layers from the model
    modified_model = Sequential([layer for layer in modified_model.layers if layer is not None])

    # Compile the modified model
    modified_model.compile(loss='categorical_crossentropy',
                            optimizer=Adam(learning_rate=0.001),
                            metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

    # Train the modified model
    history = modified_model.fit(datagen_train.flow(x_train, y_train, batch_size=batch_size),
                                  epochs=epochs,
                                  validation_data=(x_val, y_val),
                                  verbose=1)

    # Evaluate the modified model
    scores = modified_model.evaluate(x_val, y_val, verbose=1)
    y_pred = modified_model.predict(x_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')

    # Print the results of the ablation study
    print('Validation loss:', scores[0])
    print('Validation accuracy:', scores[1])
    print('Validation precision:', precision)
    print('Validation recall:', recall)
    print()

# # Load CIFAR-100 dataset
# (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
#
# # Normalize pixel values to [0, 1]
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0
#
# # Convert labels to one-hot encoding
# num_classes = 100
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)

# Combine training and validation sets
x_train_combined = np.concatenate((x_train, x_val), axis=0)
y_train_combined = np.concatenate((y_train, y_val), axis=0)

# Define the model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Print the model summary
model.summary()

# Compile the model with a categorical cross-entropy loss function and Adam optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

# Set the batch size and number of epochs
batch_size = 64
epochs = 10

# Train the model on the combined training and validation sets
history = model.fit(x_train_combined, y_train_combined,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    verbose=1)

# Evaluate the model on the test set
scores = model.evaluate(x_test, y_test, verbose=1)
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')

# Print the evaluation results
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('Test precision:', precision)
print('Test recall:', recall)


# Evaluate the model on the test set
scores = model.evaluate(x_test, y_test, verbose=1)
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')

# Print the evaluation results
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('Test precision:', precision)
print('Test recall:', recall)