import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import random

# if you're running this without the .h5 models, set this to true
training = False 
batch_size = 32 

# Calculate the number of steps (batches) in the test dataset
num_test_samples = 100
test_steps = max(1, num_test_samples // batch_size)


def visualize_dataset(dataset, class_names, num_images=9, plot_title="Sample of flowers"):
    plt.figure(figsize=(15, 15))
    plt.suptitle(plot_title, fontsize=16, y=0.93)  # Adjust the position of the main title
    for i, (image, label) in enumerate(dataset.take(num_images)):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy().astype("uint8"))
        # Adjust the position of the subtitle to be at the top inside the subplot
        plt.title(f'{label.numpy()} - {class_names[label.numpy()]}', fontsize=10, y=1.1)
        plt.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the whole layout to make space for the main title
    plt.show()

def preprocess_data(image, label, sunflower_label=3):
    """Converts labels to binary: 1 if sunflower, 0 otherwise."""
    resized_image = tf.image.resize(image, [224, 224])
    binary_label = tf.cast(tf.equal(label, sunflower_label), tf.int32)
    return resized_image, binary_label

# CORRUPT THE LABELS
def corrupt_labels(image, label, corruption_prob=0.1, num_classes=5):
    if random.random() < corruption_prob:
        # generate a random label
        corrupted_label = tf.random.uniform(shape=[], minval=0, maxval=num_classes, dtype=tf.int64)
        return image, corrupted_label
    else:
        # else we don't do nothing
        return image, label


# LOAD AND PROCESS THE INPUT
ds_full, ds_info = tfds.load(
    'tf_flowers',
    split='train',
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# features are labelled with an integer 0-4 to represent a flower
# print this to see the mapping so we can set the mapping to be a binary mapping
class_names = ds_info.features['label'].names
print(class_names)

binary_class_names = ["not sunflower", "sunflower"]



# Determine the number of samples in the dataset
num_examples = ds_info.splits['train'].num_examples
print("Total number of examples:", num_examples)

# Calculate the split sizes for 80-20 split
train_size = int(0.8 * num_examples)
test_size = num_examples - train_size
# Split the dataset into training and testing sets
ds_train = ds_full.take(train_size)
ds_test = ds_full.skip(train_size)
print("Base train set size:", len(ds_train))
print("Base test set size:", len(ds_test))

# Further reduce the size of ds_train for a proof of concept
ds_train = ds_train.take(600)
ds_test = ds_test.take(50)
print("Actual train set size:", len(ds_train))
print("Actual test set size:", len(ds_test))

binary_ds_train = ds_train.map(preprocess_data)
binary_ds_test = ds_test.map(preprocess_data)


# sanity check to see if our images make sense or not
visualize_dataset(binary_ds_train, binary_class_names, num_images=9, plot_title="training examples - uncorrupted")
visualize_dataset(binary_ds_test, binary_class_names, num_images=9, plot_title="testing examples - uncorrupted")

corrupted_ds_train_1 = binary_ds_train.map(lambda image, label: corrupt_labels(image, label, corruption_prob=0.1, num_classes=len(binary_class_names)))
corrupted_ds_train_3 = binary_ds_train.map(lambda image, label: corrupt_labels(image, label, corruption_prob=0.3, num_classes=len(binary_class_names)))
corrupted_ds_train_5 = binary_ds_train.map(lambda image, label: corrupt_labels(image, label, corruption_prob=0.5, num_classes=len(binary_class_names)))
corrupted_ds_train_7 = binary_ds_train.map(lambda image, label: corrupt_labels(image, label, corruption_prob=0.7, num_classes=len(binary_class_names)))
corrupted_ds_train_9 = binary_ds_train.map(lambda image, label: corrupt_labels(image, label, corruption_prob=0.9, num_classes=len(binary_class_names)))

visualize_dataset(corrupted_ds_train_1, binary_class_names, num_images=9, plot_title="training examples - CORRUPTED 10%")
visualize_dataset(corrupted_ds_train_3, binary_class_names, num_images=9, plot_title="training examples - CORRUPTED 30%")
visualize_dataset(corrupted_ds_train_5, binary_class_names, num_images=9, plot_title="training examples - CORRUPTED 50%")
visualize_dataset(corrupted_ds_train_7, binary_class_names, num_images=9, plot_title="training examples - CORRUPTED 70%")
visualize_dataset(corrupted_ds_train_9, binary_class_names, num_images=9, plot_title="training examples - CORRUPTED 90%")


# Apply batching and prefetching
binary_ds_train = binary_ds_train.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
binary_ds_test = binary_ds_test.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
corrupted_ds_train_1 = corrupted_ds_train_1.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
corrupted_ds_train_3 = corrupted_ds_train_3.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
corrupted_ds_train_5 = corrupted_ds_train_5.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
corrupted_ds_train_7 = corrupted_ds_train_7.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
corrupted_ds_train_9 = corrupted_ds_train_9.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)



if training:
    base_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # apparently this is better?
    base_model.trainable = False
    
    # Add custom layers on top of the base model
    control_model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    control_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    control_model.fit(binary_ds_train, epochs=5)
    control_model.save('control_binary_resnet.h5')


control_model = load_model('control_binary_resnet.h5')
loss, accuracy = control_model.evaluate(binary_ds_test, steps=test_steps)

print(f"ResNet trained on uncorrupted labels accuracy: {accuracy*100}%")
print(f"ResNet trained on uncorrupted labels loss: {loss}")






# ENTER CORRPUTING LABELS SECTION




# next, we want to train another resnet but on the corrputed labels and see the differences.
# note, training this will take around 3h30 in total 

if training:
    # trained on 0.1 probability of corrupted labels
    corrupted_resnet_1 = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        weights=None,
        classes=ds_info.features['label'].num_classes
    )
    
    # Compile the model
    corrupted_resnet_1.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Starting training ResNet with 10% corrupted labels")
    corrupted_resnet_1.fit(corrupted_ds_train_1, epochs=5)
    corrupted_resnet_1.save('resnet_corrupted_1.h5')
    
    
    # trained on 0.3 probability of corrupted labels
    corrupted_resnet_3 = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        weights=None,
        classes=ds_info.features['label'].num_classes
    )
    
    # Compile the model
    corrupted_resnet_3.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Starting training ResNet with 30% corrupted labels")
    corrupted_resnet_3.fit(corrupted_ds_train_3, epochs=5)
    corrupted_resnet_3.save('resnet_corrupted_3.h5')
    
    
    # trained on 0.5 probability of corrupted labels
    corrupted_resnet_5 = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        weights=None,
        classes=ds_info.features['label'].num_classes
    )
    
    # Compile the model
    corrupted_resnet_5.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Starting training ResNet with 50% corrupted labels")
    corrupted_resnet_5.fit(corrupted_ds_train_5, epochs=5)
    corrupted_resnet_5.save('resnet_corrupted_5.h5')
    
    # trained on 0.7 probability of corrupted labels
    corrupted_resnet_7 = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        weights=None,
        classes=ds_info.features['label'].num_classes
    )
    
    # Compile the model
    corrupted_resnet_7.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Starting training ResNet with 70% corrupted labels")
    corrupted_resnet_7.fit(corrupted_ds_train_7, epochs=5)
    corrupted_resnet_7.save('resnet_corrupted_7.h5')
    
    
    # trained on 0.9 probability of corrupted labels
    corrupted_resnet_9 = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        weights=None,
        classes=ds_info.features['label'].num_classes
    )
    
    # Compile the model
    corrupted_resnet_9.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Starting training ResNet with 90% corrupted labels")
    corrupted_resnet_9.fit(corrupted_ds_train_9, epochs=5)
    corrupted_resnet_9.save('resnet_corrupted_9.h5')


corrupted_resnet_1 = load_model('resnet_corrupted_1.h5')
corrupted_resnet_3 = load_model('resnet_corrupted_3.h5')
corrupted_resnet_5 = load_model('resnet_corrupted_5.h5')
corrupted_resnet_7 = load_model('resnet_corrupted_7.h5')
corrupted_resnet_9 = load_model('resnet_corrupted_9.h5')

cr1_loss, cr1_accuracy = corrupted_resnet_1.evaluate(binary_ds_test, steps=test_steps)
cr3_loss, cr3_accuracy = corrupted_resnet_3.evaluate(binary_ds_test, steps=test_steps)
cr5_loss, cr5_accuracy = corrupted_resnet_5.evaluate(binary_ds_test, steps=test_steps)
cr7_loss, cr7_accuracy = corrupted_resnet_7.evaluate(binary_ds_test, steps=test_steps)
cr9_loss, cr9_accuracy = corrupted_resnet_9.evaluate(binary_ds_test, steps=test_steps)

print(f"Resnet trained on corrupted labels (10% chance) test accuracy: {cr1_accuracy*100}%")
print(f"ResNet trained on corrupted labels (30% chance) test accuracy: {cr3_accuracy*100}%")
print(f"ResNet trained on corrupted labels (50% chance) test accuracy: {cr5_accuracy*100}%")
print(f"ResNet trained on corrupted labels (70% chance) test accuracy: {cr7_accuracy*100}%")
print(f"ResNet trained on corrupted labels (90% chance) test accuracy: {cr9_accuracy*100}%")


print(f"ResNet trained on corrupted labels (10% chance) test loss: {cr1_loss}")
print(f"ResNet trained on corrupted labels (30% chance) test loss: {cr3_loss}")
print(f"ResNet trained on corrupted labels (50% chance) test loss: {cr5_loss}")
print(f"ResNet trained on corrupted labels (70% chance) test loss: {cr7_loss}")
print(f"ResNet trained on corrupted labels (90% chance) test loss: {cr9_loss}")




# ROBUST METHOD
# use L1 loss instead of binary cross entropy?

if training:
     # trained on 0.1 probability of corrupted labels
     l1_corrupted_resnet_1 = tf.keras.applications.ResNet50(
         input_shape=(224, 224, 3),
         weights=None,
         classes=ds_info.features['label'].num_classes
     )
     
     # Compile the model with L1 loss
     l1_corrupted_resnet_1.compile(
         optimizer='adam',
         loss='mean_absolute_error',  # Using L1 loss
         metrics=['accuracy']
     )
     
     # Train the model
     print("Starting training ResNet with 10% corrupted labels - WITH L1 NORM")
     history_l1_cr1 = l1_corrupted_resnet_1.fit(corrupted_ds_train_1, epochs=5)
     l1_corrupted_resnet_1.save('l1_resnet_corrupted_1.h5')
     
     
     
     # trained on 0.3 probability of corrupted labels
     l1_corrupted_resnet_3 = tf.keras.applications.ResNet50(
         input_shape=(224, 224, 3),
         weights=None,
         classes=ds_info.features['label'].num_classes
     )
     
     # Compile the model with L1 loss
     l1_corrupted_resnet_3.compile(
         optimizer='adam',
         loss='mean_absolute_error',  # Using L1 loss
         metrics=['accuracy']
     )
     
     # Train the model
     print("Starting training ResNet with 30% corrupted labels - WITH L1 NORM")
     history_l1_cr3 = l1_corrupted_resnet_3.fit(corrupted_ds_train_3, epochs=5)
     l1_corrupted_resnet_3.save('l1_resnet_corrupted_3.h5')
     
     
     # trained on 0.5 probability of corrupted labels
     l1_corrupted_resnet_5 = tf.keras.applications.ResNet50(
         input_shape=(224, 224, 3),
         weights=None,
         classes=ds_info.features['label'].num_classes
     )
     
     # Compile the model with L1 loss
     l1_corrupted_resnet_5.compile(
         optimizer='adam',
         loss='mean_absolute_error',  # Using L1 loss
         metrics=['accuracy']
     )
     
     # Train the model
     print("Starting training ResNet with 50% corrupted labels - WITH L1 NORM")
     history_l1_cr5 = l1_corrupted_resnet_5.fit(corrupted_ds_train_5, epochs=5)
     l1_corrupted_resnet_5.save('l1_resnet_corrupted_5.h5')
     
     
     
     # trained on 0.7 probability of corrupted labels
     l1_corrupted_resnet_7 = tf.keras.applications.ResNet50(
         input_shape=(224, 224, 3),
         weights=None,
         classes=ds_info.features['label'].num_classes
     )
     
     # Compile the model with L1 loss
     l1_corrupted_resnet_7.compile(
         optimizer='adam',
         loss='mean_absolute_error',  # Using L1 loss
         metrics=['accuracy']
     )
     
     # Train the model
     print("Starting training ResNet with 70% corrupted labels - WITH L1 NORM")
     history_l1_cr7 = l1_corrupted_resnet_7.fit(corrupted_ds_train_7, epochs=5)
     l1_corrupted_resnet_7.save('l1_resnet_corrupted_7.h5')
     
     
     
     # trained on 0.9 probability of corrupted labels
     l1_corrupted_resnet_9 = tf.keras.applications.ResNet50(
         input_shape=(224, 224, 3),
         weights=None,
         classes=ds_info.features['label'].num_classes
     )
     
     # Compile the model with L1 loss
     l1_corrupted_resnet_9.compile(
         optimizer='adam',
         loss='mean_absolute_error',  # Using L1 loss
         metrics=['accuracy']
     )
     
     # Train the model
     print("Starting training ResNet with 90% corrupted labels - WITH L1 NORM")
     history_l1_cr9 = l1_corrupted_resnet_9.fit(corrupted_ds_train_9, epochs=5)
     l1_corrupted_resnet_9.save('l1_resnet_corrupted_9.h5')
     








l1_corrupted_resnet_1 = load_model('l1_resnet_corrupted_1.h5')
l1_corrupted_resnet_3 = load_model('l1_resnet_corrupted_3.h5')
l1_corrupted_resnet_5 = load_model('l1_resnet_corrupted_5.h5')
l1_corrupted_resnet_7 = load_model('l1_resnet_corrupted_7.h5')
l1_corrupted_resnet_9 = load_model('l1_resnet_corrupted_9.h5')

l1_cr1_loss, l1_cr1_accuracy = l1_corrupted_resnet_1.evaluate(binary_ds_test, steps=test_steps)
l1_cr3_loss, l1_cr3_accuracy = l1_corrupted_resnet_3.evaluate(binary_ds_test, steps=test_steps)
l1_cr5_loss, l1_cr5_accuracy = l1_corrupted_resnet_5.evaluate(binary_ds_test, steps=test_steps)
l1_cr7_loss, l1_cr7_accuracy = l1_corrupted_resnet_7.evaluate(binary_ds_test, steps=test_steps)
l1_cr9_loss, l1_cr9_accuracy = l1_corrupted_resnet_9.evaluate(binary_ds_test, steps=test_steps)

print(f"l1-resnet trained on corrupted labels (10% chance) test accuracy: {l1_cr1_accuracy*100}%")
print(f"l1-resnet trained on corrupted labels (30% chance) test accuracy: {l1_cr3_accuracy*100}%")
print(f"l1-resnet trained on corrupted labels (50% chance) test accuracy: {l1_cr5_accuracy*100}%")
print(f"l1-resnet trained on corrupted labels (70% chance) test accuracy: {l1_cr7_accuracy*100}%")
print(f"l1-resnet trained on corrupted labels (90% chance) test accuracy: {l1_cr9_accuracy*100}%")

print(f"l1-resnet trained on corrupted labels (10% chance) test loss: {l1_cr1_loss}")
print(f"l1-resnet trained on corrupted labels (30% chance) test loss: {l1_cr3_loss}")
print(f"l1-resnet trained on corrupted labels (50% chance) test loss: {l1_cr5_loss}")
print(f"l1-resnet trained on corrupted labels (70% chance) test loss: {l1_cr7_loss}")
print(f"l1-resnet trained on corrupted labels (90% chance) test loss: {l1_cr9_loss}")
