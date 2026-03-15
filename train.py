import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

print("Starting training...")

IMG_SIZE = (150, 150)
BATCH = 32

train_data = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.1)
test_data  = ImageDataGenerator(rescale=1./255)

train_gen = train_data.flow_from_directory("chest_xray/train", target_size=IMG_SIZE, batch_size=BATCH, class_mode="binary")
val_gen   = test_data.flow_from_directory("chest_xray/val",   target_size=IMG_SIZE, batch_size=BATCH, class_mode="binary")
test_gen  = test_data.flow_from_directory("chest_xray/test",  target_size=IMG_SIZE, batch_size=BATCH, class_mode="binary")

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_gen, validation_data=val_gen, epochs=10)

loss, acc = model.evaluate(test_gen)
print(f"\nTest Accuracy: {acc*100:.2f}%")

model.save("pneumonia_model.h5")
print("Model saved as pneumonia_model.h5 ✅")