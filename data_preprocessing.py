import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image data generator for augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load training data
train_generator = datagen.flow_from_directory(
    'pest_images/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Load validation data
validation_generator = datagen.flow_from_directory(
    'pest_images/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Save generators for later use
import pickle
with open('train_generator.pkl', 'wb') as f:
    pickle.dump(train_generator, f)
with open('validation_generator.pkl', 'wb') as f:
    pickle.dump(validation_generator, f)
