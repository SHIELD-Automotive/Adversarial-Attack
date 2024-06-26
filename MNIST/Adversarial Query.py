import numpy as np

def adversarial_query(image):
    # Introduce small perturbations
    noise = np.random.normal(loc=0.0, scale=0.1, size=image.shape)
    adversarial_image = np.clip(image + noise, 0., 1.)
    return adversarial_image

# Assume `model` is your trained MNIST model
# Assume `test_images` are your preprocessed MNIST test images

# Select a single image to attack
image_to_attack = test_images[0:1]

# Create an adversarial image
adversarial_image = adversarial_query(image_to_attack)

# Use the model to predict the class of the adversarial image
prediction = model.predict(adversarial_image)
predicted_class = np.argmax(prediction, axis=1)
print(f"Predicted class for adversarial image: {predicted_class}")
