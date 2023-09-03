import pandas as pd
import os
import random
import shutil


# TRAIN FILES SETUP

# Set the paths to your dataset folders
dataset_dir = r"C:\Users\abuth\Downloads\Casia\CASIA2"
real_dir = os.path.join(dataset_dir, "REAL","train")
fake_dir = os.path.join(dataset_dir, "FAKE","train")

# Set the paths to the new directories that will contain the selected images
train_dir = r"C:\Users\abuth\OneDrive\Documents\Thesis-Project\Dataset1\train"
real_train_dir = os.path.join(train_dir, "REAL")
fake_train_dir = os.path.join(train_dir, "FAKE")

# Create the new directories if they don't exist
if not os.path.exists(real_train_dir):
    os.makedirs(real_train_dir)
if not os.path.exists(fake_train_dir):
    os.makedirs(fake_train_dir)

# Set the number of images to select from each folder
num_images = 2000

# Randomly select the required number of images from the REAL folder and copy them to the new directory
real_images = os.listdir(real_dir)
selected_real_images = random.sample(real_images, num_images)
for image_name in selected_real_images:
    source_path = os.path.join(real_dir, image_name)
    dest_path = os.path.join(real_train_dir, image_name)
    shutil.copyfile(source_path, dest_path)

# Randomly select the required number of images from the FAKE folder and copy them to the new directory
fake_images = os.listdir(fake_dir)
selected_fake_images = random.sample(fake_images, num_images)
for image_name in selected_fake_images:
    source_path = os.path.join(fake_dir, image_name)
    dest_path = os.path.join(fake_train_dir, image_name)
    shutil.copyfile(source_path, dest_path)
    
    
# TEST FILES SETUP

# Set the paths to your dataset folders
dataset_dir_test = r"C:\Users\abuth\Downloads\Casia\CASIA2"
real_dir = os.path.join(dataset_dir_test, "REAL","train")
fake_dir = os.path.join(dataset_dir_test, "FAKE","train")
# Set the paths to the new directories that will contain the selected images
test_dir = r"C:\Users\abuth\OneDrive\Documents\Thesis-Project\Dataset1\test"
real_test_dir = os.path.join(test_dir, "REAL")
fake_test_dir = os.path.join(test_dir, "FAKE")

# Create the new directories if they don't exist
if not os.path.exists(real_test_dir):
    os.makedirs(real_test_dir)
if not os.path.exists(fake_test_dir):
    os.makedirs(fake_test_dir)

# Set the number of images to select from each folder
num_images = 200

# Randomly select the required number of images from the REAL folder and copy them to the new directory
real_images = os.listdir(real_dir)
selected_real_images = random.sample(real_images, num_images)
for image_name in selected_real_images:
    source_path = os.path.join(real_dir, image_name)
    dest_path = os.path.join(real_test_dir, image_name)
    shutil.copyfile(source_path, dest_path)

# Randomly select the required number of images from the FAKE folder and copy them to the new directory
fake_images = os.listdir(fake_dir)
selected_fake_images = random.sample(fake_images, num_images)
for image_name in selected_fake_images:
    source_path = os.path.join(fake_dir, image_name)
    dest_path = os.path.join(fake_test_dir, image_name)
    shutil.copyfile(source_path, dest_path)