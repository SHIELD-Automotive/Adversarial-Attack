#Below is an example code snippet that adds Gaussian noise to a single test image from the MNIST dataset and then uses the previously defined model to predict the noisy image's label
#If you want the noise to have a standard deviation (std) greater than 1, you can simply adjust the std parameter in the add_gaussian_noise function.

import numpy as np

def add_gaussian_noise(image, mean=0.0, std=0.1):
    """
    Adds Gaussian noise to an image.
    
    Parameters:
    - image: numpy array of shape (height, width, channels)
    - mean: mean of the Gaussian noise
    - std: standard deviation of the Gaussian noise
    
    Returns:
    - noisy_image: image with Gaussian noise added
    """
    gaussian_noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 1)  # Clip the image to be between 0 and 1
    return noisy_image

# Select a single test image and label
test_image = test_images[1]  # Assuming this is already scaled to [0, 1]
test_label = test_labels[1]

# Add Gaussian noise to the test image
noisy_test_image = add_gaussian_noise(test_image)

# Display the original and noisy images, if you want to visually compare them
# This code is meant to run in a Jupyter notebook or a similar environment where you can display images.
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(test_image.squeeze(), cmap='gray')  # Assuming the image is in grayscale
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Noisy Image")
plt.imshow(noisy_test_image.squeeze(), cmap='gray')
plt.axis('off')

plt.show()

# Use the model to predict the label of the noisy image
noisy_test_image_expanded = np.expand_dims(noisy_test_image, axis=0)  # Add batch dimension
prediction = model.predict(noisy_test_image_expanded)
predicted_label = np.argmax(prediction, axis=1)

print('Predicted label for the noisy image:', predicted_label)
