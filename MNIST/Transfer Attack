def generate_adversarial_example(model, image, label):
    image = tf.Variable(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    adversarial_image = image + 0.1 * signed_grad
    return tf.clip_by_value(adversarial_image, 0, 1)

# Assume `source_model` and `target_model` are your two MNIST models
# Assume `test_images` and `test_labels` are your preprocessed test images and labels

# Select a single image and label
image = test_images[0:1]
label = test_labels[0:1]

# Generate an adversarial example using the source model
adversarial_image = generate_adversarial_example(source_model, image, label)

# Evaluate the adversarial example on the target model
prediction = target_model.predict(adversarial_image)
predicted_class = np.argmax(prediction, axis=1)
print(f"Predicted class for adversarial image on target model: {predicted_class}")
