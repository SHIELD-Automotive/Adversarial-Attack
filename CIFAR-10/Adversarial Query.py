def adversarial_query(image, epsilon=0.01):
    # Add small perturbations to create an adversarial example
    perturbation = epsilon * np.sign(np.random.normal(size=image.shape))
    adversarial_image = np.clip(image + perturbation, 0, 1)
    return adversarial_image

# Assume `model` is your trained CIFAR-10 model
# Assume `test_images` are your preprocessed CIFAR-10 test images

# Select a single image to attack : image_to_attack = test_images[0:1]
#or
#Select a random test image:
index = np.random.randint(0, len(test_images))
image_to_attack = test_images[index]
true_label = np.argmax(test_labels[index])

# Create an adversarial image
adversarial_image = adversarial_query(image_to_attack)

# Use the model to predict the class of the adversarial image
prediction = model.predict(adversarial_image)
predicted_class = np.argmax(prediction, axis=1)
print(f"Predicted class for adversarial image: {predicted_class}")

# Display the original and adversarial images with their labels
display_images(image_to_attack, adversarial_image, true_label, adversarial_class)
