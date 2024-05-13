"""
Module to prepare dataset by using randomly augmenting images
of all classes and combining augmented and original dataset
"""

import os
import shutil

import cv2
import albumentations as A
from sklearn.model_selection import train_test_split

from base_logger import logger

ORIGINAL_DATASET = 'dataset/original'  # original dataset path
AUGMENTED_DATASET = 'dataset/augmented'  # dataset after some prerpocessing
# final dataset with train-validationtest-test split
FINAL_DATASET = 'dataset/final'

# get list of document classes in dataset
CLASSES = os.listdir(ORIGINAL_DATASET)

# image dimension
IMG_HEIGHT = 224
IMG_WIDTH = 224

# image transformation pipeline for augmentation
transform = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.Affine(shear=[-20, 20], p=0.2),
        A.RandomBrightnessContrast(),
        A.ToGray(p=0.3)
    ]
)


def augment_image(image_path, output_file):
    """
    Augments an image using the defined transformation chain and saves it to
    the specified output path.

    Args:
      image_path: Path to the image file to be augmented.
      output_file: Path to the file where the augmented image will be saved.
    """
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image=image)
        print(f'Augmenting {image_path} to {output_file}')
        cv2.imwrite(output_file, image['image'])

    except FileNotFoundError:
        print(f'Image {image_path} not found, skipping')
    except Exception as e:
        print('SOmething went wrong', e)


def augment_dataset(dataset_path, output_path):
    """
    Augments all images in a dataset directory and saves them to a separate
    directory with class subdirectories.

    Args:
      dataset_path: Path to the directory containing the original dataset.
      output_path: Path to the directory where the augmented dataset will
      be saved.
    """
    for doc_class in CLASSES:
        class_path = os.path.join(dataset_path, doc_class)

        images = os.listdir(class_path)
        logger.info('Augmenting %i of class %s', len(images), doc_class)
        class_output_path = os.path.join(output_path, doc_class)

        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)

        for img in images:
            img_path = os.path.join(class_path, img)
            output_file = os.path.join(class_output_path, 'aug_'+img)
            augment_image(img_path, output_file)


def merge_dataset(original_data, augmented_data, final_data):
    """
    Merges the original dataset with the augmented dataset into
    a single final dataset.This function combines the images from
    the original dataset and the augmented dataset, placing them
    together in a new directory structure that mirrors the
    class subdirectories of the original dataset.

    Args:
        original_data: Path to the directory containing the original dataset.
        augmented_data: Path to the directory containing the augmented dataset.
        final_data: Path to the directory where the merged final dataset
        will be saved.

    Raises:
        FileNotFoundError: If either the original or augmented dataset
        directory is not found.
        Exception: For any other errors encountered during the merging process.
    """

    try:

        # Check if source and destination dataset directories exist
        if not os.path.exists(original_data):
            raise FileNotFoundError('Original dataset not found')
        if not os.path.exists(augmented_data):
            raise FileNotFoundError('Augmented dataset not found')
        if not os.path.exists(final_data):
            # create destination directory if not exist
            os.makedirs(final_data)
            os.makedirs(os.path.join(final_data, 'train'))
            os.makedirs(os.path.join(final_data, 'val'))

        for doc_class in CLASSES:
            final_path_train = os.path.join(final_data, 'train', doc_class)
            final_path_val = os.path.join(final_data, 'val', doc_class)

            if not os.path.exists(final_path_train):
                os.makedirs(final_path_train)

            if not os.path.exists(final_path_val):
                os.makedirs(final_path_val)

            augmented_path = os.path.join(augmented_data, doc_class)
            original_path = os.path.join(original_data, doc_class)

            original_images = [os.path.join(
                original_path, path) for path in os.listdir(original_path)]
            augmented_images = [os.path.join(
                augmented_path, path) for path in os.listdir(augmented_path)]
            print(f'\nClass: {doc_class}')
            print(f'Original images: {len(original_images)}')
            print(f'Augmented images: {len(augmented_images)}')

            all_images = original_images + augmented_images
            train_images, val_images = train_test_split(
                all_images, test_size=0.3, random_state=42)

            print(f'Train images: {len(train_images)}')
            print(f'Val images: {len(val_images)}')

            logger.info('Splitting class %s into train and val set', doc_class)
            for image in train_images:
                shutil.copyfile(image,
                                os.path.join(final_path_train,
                                             image.split('/')[-1]))
            for image in val_images:
                shutil.copyfile(image,
                                os.path.join(final_path_val,
                                             image.split('/')[-1]))

        # Comment out the below line if you want
        #  to keep the augmented dataset directory
        # shutil.rmtree(augmented_data)
        logger.info(
            'Original and augmented images combined to a single dataset at %s', final_data)
    except FileNotFoundError:
        logger.error('Unable to locate files to merge data')
    except Exception:
        logger.error('Failed to merge original and augmented dataset')


def prepare_dataset():
    '''
    Function to prepare dataset with following steps
        1. Apply image transforms for augmentation
        2. Merge Augmented and original dataset
        3. Split merged dataset into train and validation set
    '''
    # Augment Original Images and save
    augment_dataset(ORIGINAL_DATASET, AUGMENTED_DATASET)

    # merge augmented and original image sets
    merge_dataset(ORIGINAL_DATASET, AUGMENTED_DATASET, FINAL_DATASET)


if __name__ == '__main__':
    prepare_dataset()
