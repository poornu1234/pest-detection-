import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pickle

# Load data generators
with open('train_generator.pkl', 'rb') as f:
    train_generator = pickle.load(f)
with open('validation_generator.pkl', 'rb') as f:
    validation_generator = pickle.load(f)

# Load the InceptionV3 model pre-trained on ImageNet
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add new layers for our specific task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(9, activation='softmax')(x)  # 9 classes for 9 types of pests

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=20, validation_data=validation_generator)

# Save the trained model
model.save('pest_detection_model.h5')
