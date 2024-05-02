import numpy as np
import matplotlib.pyplot as plt

# Your provided matrix
matrix_a = np.array([[1,0,0,0,1],
                     [0,1,0,1,0],
                     [0,0,1,0,0],
                     [0,1,0,1,0],
                     [1,0,0,0,1]])

# Normalize the matrix to have the same scale as the MNIST images
matrix_a = matrix_a.astype('float32')

# Assuming test_images is normalized and has shape (num_samples, 28, 28, 1)
# Select a single test image
test_image = test_images[1].squeeze()  # Remove the channel dimension for simplicity

# Copy the test image to avoid altering the original image
modified_test_image = np.copy(test_image)

# Add the matrix pattern to the upper-left corner of the test image
# Note: The MNIST images and your matrix are normalized (values between 0 and 1)
for i in range(matrix_a.shape[0]):
    for j in range(matrix_a.shape[1]):
        # This operation adds the matrix on top of the image, potentially increasing intensity
        modified_test_image[i, j] += matrix_a[i, j]
        # Ensure the values are still between 0 and 1 after addition
        modified_test_image[i, j] = min(modified_test_image[i, j], 1.0)

# Display the original and modified images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(test_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Modified Image with Matrix")
plt.imshow(modified_test_image, cmap='gray')
plt.axis('off')

plt.show()
