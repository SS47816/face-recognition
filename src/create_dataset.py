import os
import random
import shutil
import numpy as np
import cv2

def splitPIEDataset(repo_path):

    # Set destination paths
    dataset_path = os.path.join(repo_path, 'data/PIE')
    train_set_path = os.path.join(repo_path, 'data/train')
    test_set_path = os.path.join(repo_path, 'data/test')
    # Remove old folders at the destination (if any)
    try:
        shutil.rmtree(train_set_path)
        shutil.rmtree(test_set_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    # Create destination folders
    os.mkdir(train_set_path)
    os.mkdir(test_set_path)

    # Select 25 from the 68 subjects
    selected_subjects = random.sample(os.listdir(dataset_path), 25)

    # Within each subject, split into 70-30 train-test set
    for subject_path in selected_subjects:
        if subject_path == '.DS_Store':
            continue
        img_paths = os.listdir(os.path.join(dataset_path, subject_path))
        
        # Split img file into train/test sets
        train_set = random.sample(img_paths, int(len(img_paths)*0.7))
        test_set = [item for item in img_paths if item not in train_set]
        
        # Copy files to new dataset folders
        src = os.path.join(dataset_path, subject_path)
        
        train_subject_path = os.path.join(train_set_path, subject_path)
        os.mkdir(train_subject_path)
        for img_file in train_set:
            print(os.path.join(src, img_file))
            print(train_subject_path)
            shutil.copy2(os.path.join(src, img_file), train_subject_path)
        
        test_subject_path = os.path.join(test_set_path, subject_path)
        os.mkdir(test_subject_path)
        for img_file in test_set:
            print(os.path.join(src, img_file))
            print(test_subject_path)
            shutil.copy2(os.path.join(src, img_file), test_subject_path)

def main():
    # Define paths to the original PIE dataset (please modify)
    repo_path = '/home/ss/ss_ws/face-recognition'

    # 

    # Split the PIE dataset into train and test sets
    # splitPIEDataset(repo_path)

if __name__ == "__main__":
    main()