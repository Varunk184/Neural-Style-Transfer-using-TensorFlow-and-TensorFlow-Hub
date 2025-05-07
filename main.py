import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import os

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
    
    img = PIL.Image.open(path_to_img)
    img = np.array(img.convert('RGB'))
    img = img / 255.0 
    
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


content_path = r'D:\neural_style_transfer_project\cat.png'
style_path = r'D:\neural_style_transfer_project\nightimage.png'

print(f"Loading content image from: {content_path}")
content_image = load_img(content_path)

print(f"Loading style image from: {style_path}")
style_image = load_img(style_path)

# Use TF Hub for the model
import tensorflow_hub as hub
print("Loading model from TensorFlow Hub...")
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

print("Applying style transfer...")
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# Save output
output_path = "output.jpg"
print(f"Saving result to: {output_path}")
tensor_to_image(stylized_image).save(output_path)

# Display output
plt.figure(figsize=(12, 12))
plt.imshow(tensor_to_image(stylized_image))
plt.axis('off')
plt.show()

print("Style transfer completed successfully!")