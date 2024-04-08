import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical

# Ensure you've loaded and preprocessed your MNIST data as before
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Define FGSM attack
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = tf.sign(data_grad)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# Create a loss object
loss_object = keras.losses.CategoricalCrossentropy()

# Select a sample to perturb
image = test_images[0:1]
image_label = test_labels[0:1]

# Record the gradients of the image with respect to the loss
with tf.GradientTape() as tape:
    tape.watch(image)
    prediction = model(image)
    loss = loss_object(image_label, prediction)

# Get the gradients of the loss w.r.t to the input image.
gradient = tape.gradient(loss, image)

# Get the sign of the gradients to create the perturbation
epsilon = 0.1  # This is the magnitude of the noise
perturbed_image = fgsm_attack(image, epsilon, gradient)

# Re-evaluate the model on the perturbed image
perturbed_pred = model(perturbed_image)
perturbed_label = tf.argmax(perturbed_pred, axis=1)
original_label = tf.argmax(image_label, axis=1)

print('Original label:', original_label.numpy())
print('Predicted label for perturbed image:', perturbed_label.numpy())

#epsilon controls the magnitude of the attack; i.e., how much noise is added. 
#The fgsm_attack function creates a new image that's slightly different from the original. 
#This new image is then fed into the model to see if the prediction changes.
