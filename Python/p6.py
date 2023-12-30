import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


labels_to_include=['all', 'must', 'never', 'none', 'only', 'up', 'down', 'yes', 'no']
def calculate_max_length(directory, labels_to_include, window_size, stride, n_mels):
    max_length = 0
    for label in labels_to_include:
        label_dir = os.path.join(directory, label)
        for file in os.listdir(label_dir):
            if file.lower().endswith('.wav'):
                file_path = os.path.join(label_dir, file)
                audio, sr = librosa.load(file_path, sr=None)
                hop_length = int(sr * stride)
                n_fft = int(sr * window_size)
                S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
                if S.shape[1] > max_length:
                    max_length = S.shape[1]
    return max_length



def load_data(directory, labels_to_include, window_size=0.03, stride=0.02, n_mels=40, max_length=None):   
    labels = []
    spectrograms = []
    if max_length is None:
        max_length = calculate_max_length(directory, labels_to_include, window_size=0.03, stride=0.02, n_mels=40)

    for label in os.listdir(directory):
        if label in labels_to_include:
            label_dir = os.path.join(directory, label)
            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                audio, sr = librosa.load(file_path, sr=None)
                hop_length = int(sr * stride)
                n_fft = int(sr * window_size)
                S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
                log_S = librosa.power_to_db(S, ref=np.max)

                if log_S.shape[1] < max_length:
                    pad_width = max_length - log_S.shape[1]
                    log_S = np.pad(log_S, pad_width=((0, 0), (0, pad_width)), mode='constant')

                spectrograms.append(log_S)
                labels.append(label)

    return np.array(spectrograms), np.array(labels)

# Usage
directory = '/home/tulasi/eml/P6/data_speech_commands_v0.02/'
labels_to_include = ['all', 'must', 'never', 'none', 'only', 'up', 'down', 'yes', 'no']
max_length = calculate_max_length(directory, labels_to_include, window_size=0.03, stride=0.02, n_mels=40)
spectrograms, labels = load_data(directory, labels_to_include, max_length=max_length)

#data preperation
spectrogram_shape = spectrograms[0].shape
num_classes = len(set(labels))

label_mapping = {label: idx for idx, label in enumerate(set(labels))}
numerical_labels = np.array([label_mapping[label] for label in labels])

spectrograms_reshaped = spectrograms.reshape(spectrograms.shape[0], spectrogram_shape[0], spectrogram_shape[1], 1)


def create_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Splitting the data into training and testing sets
train_spectrograms, test_spectrograms, train_labels, test_labels = train_test_split(
    spectrograms_reshaped, numerical_labels, test_size=0.2, random_state=42)

# Create the model
model = create_model(spectrogram_shape + (1,), num_classes)

# Training the model and saving the history
history = model.fit(train_spectrograms, train_labels, epochs=10, validation_data=(test_spectrograms, test_labels))

# Plotting accuracy and loss graphs
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_spectrograms, test_labels, verbose=2)
# Print the model summary
model.summary()
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict and calculate per-class accuracies
predictions = model.predict(test_spectrograms)
predicted_classes = np.argmax(predictions, axis=1)
confusion_mtx = tf.math.confusion_matrix(test_labels, predicted_classes)

# Calculate per-class accuracies
per_class_accuracy = tf.linalg.diag_part(confusion_mtx) / tf.reduce_sum(confusion_mtx, axis=1)
for label, acc in zip(label_mapping.keys(), per_class_accuracy):
    print(f"Accuracy for class {label}: {acc.numpy() * 100:.2f}%")


# Save the model
model.save("my_model")

# Load the saved model
model = tf.keras.models.load_model("my_model")

# Set up the converter with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_quantized_model = converter.convert()

# Save the quantized model
with open("micro_features_model.tflite", "wb") as f:
    f.write(tflite_quantized_model)

print("ALL done")


model_size = os.path.getsize("my_model") / (1024 * 1024)  # Size in MB
print(f"Model Size: {model_size:.2f} MB")

