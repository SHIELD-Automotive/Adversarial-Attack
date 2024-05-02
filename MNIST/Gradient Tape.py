import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, batch_size=64, epochs=3, validation_data=(x_test, y_test))

# Adversarial attack function
def adversarial_pattern(image, label, model):
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.categorical_crossentropy(label, prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    return signed_grad

# Generate an adversarial example
epsilon = 0.1
image = x_train[0:1] # Using the first image in the dataset
label = y_train[0:1]
perturbations = adversarial_pattern(image, label, model)
adv_example = image + epsilon * perturbations

# Compare predictions
original_pred = model.predict(image)
adv_pred = model.predict(adv_example)

#show the original and modified model
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Tape")
plt.imshow(original_pred, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Modified Tape")
plt.imshow(adv_pred, cmap='gray')
plt.axis('off')

plt.show()
