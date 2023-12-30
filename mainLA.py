import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_io as tfio  # Import tensorflow-io for signal processing





training_dir = pathlib.Path("W:\\workdir2\\CQT\\train")
validation_dir = pathlib.Path("W:\\workdir2\\CQT\\dev")
test_dir = pathlib.Path("W:\\workdir2\\CQT\\test")

test_count = len(list(test_dir.glob('*/*.png')))
train_count = len(list(training_dir.glob('*/*.png')))
#validation_split = test_count / train_count
# Define parameters and create datasets
batch_size = 50
epochs = 100
img_height = 224
img_width = 224


print("\033[1mCreating training and validation datasets:\033[0m")
training_ds = tf.keras.utils.image_dataset_from_directory(
    training_dir,
    validation_split=None,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    crop_to_aspect_ratio=True,
    label_mode='binary',
    class_names=['fake', 'real']
)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'  # You can explore more options in the documentation
)
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    seed=123
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    validation_split=None,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    crop_to_aspect_ratio=True,
    label_mode='binary',
    class_names=['fake', 'real']
)

print("\n\033[1mCreating test dataset:\033[0m")
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    crop_to_aspect_ratio=True,
    label_mode='binary',
    class_names=['fake', 'real']
)

class_names = training_ds.class_names
print("\nNames of", str(len(class_names)), "classes:", class_names)


# Build the model
base_model = keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    pooling="avg"
)
print("number of layers:", len(base_model.layers))
'''
# Freeze layers except the last few
for layer in base_model.layers[:-30]:  # Unfreeze the last 7 layers for example
    layer.trainable = False
'''
for layer in base_model.layers[:-20]:
    # Unfreeze all layers for training from scratch
    layer.trainable = False
# Create your model on top of the base model
model = keras.Sequential([
    base_model,
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))
])
#model.add(keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l1(0.01)))
#model.layers[0].trainable = True

# Define the optimizer with a specific learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(),
        keras.metrics.Precision(),
        keras.metrics.Recall(),
        keras.metrics.TruePositives(),
        keras.metrics.FalsePositives(),
        keras.metrics.TrueNegatives(),
        keras.metrics.FalseNegatives(),
    ]
)

model.summary()

# Model training
steps_per_epoch = len(training_ds) / epochs

t0 = time.time()

# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_ds,
    validation_steps=1,
    epochs=epochs
)

# Access training loss and accuracy
train_loss = history.history['loss']
train_accuracy = history.history['binary_accuracy']
val_loss = history.history['val_loss']  # Validation loss
val_accuracy = history.history['val_binary_accuracy']  # Validation accuracy

# Print or use the values as needed
print("----------------------------------------")
print("Training Loss per epoch:", train_loss)
print("Training Accuracy per epoch:", train_accuracy)
print("Validation Loss per epoch:", val_loss)
print("----------------------------------------")

t1 = time.time()
dt = (t1 - t0)

# Model evaluation
results = model.evaluate(
    test_ds,
    return_dict=True
)
test_accuracy = results['binary_accuracy']
print("Test Accuracy", test_accuracy )
model.save("W:/workdir/Models/model_LA1.h5")

plt.figure(figsize=(20, 10))

# Plotting training loss
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.suptitle('Resnet50 Trained on FoR Dataset with 20 frozen layers- ImageNet Weights', fontsize=16, y=1.02)
plt.savefig("W:/workdir/Plots/plot_LA.png")
plt.show()


def f1score(p, r):
    epsilon = 1e-7  # A small value to avoid division by zero

    # Handling potential division by zero cases
    if p == 0 and r == 0:
        return 0.0  # Return 0 when both precision and recall are zero
    elif p + r == 0:
        return 0.0  # Return 0 when the sum of precision and recall is zero
    else:
        f1 = 2 * p * r / (p + r + epsilon)
        return f1

print("-" * 70)
print('\033[1m' + "Model metrics:" + '\033[0m')
for i in results:
    print(i + ": " + str(results[i]))
print("-" * 70)
print("F1 Score: " + str(f1score(results['precision'], results['recall'])))
print("Time to train: ", dt)
print("-" * 70)

tp, fp = results['true_positives'], results['false_positives']
fn, tn = results['false_negatives'], results['true_negatives']
cmx = np.array([[tp, fp], [fn, tn]], np.int32)



cmx_plot = sns.heatmap(
    cmx / np.sum(cmx),
    cmap='Blues',
    annot=True,
    fmt=".1%",
    linewidth=5,
    cbar=False,
    square=True,
    xticklabels=['Spoof (1)', 'Real (0)'],
    yticklabels=['Spoof (1)', 'Real (0)']
)
cmx_plot.set(xlabel="Actual", ylabel="Predicted")


