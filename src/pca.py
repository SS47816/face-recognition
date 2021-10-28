import os
import math
import numpy as np
import cv2
import torch

def getPCAResults(imgs, sort=True):
    """
    Compute PCA based on the input img.

    Returns
    -------
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    mean: np.ndarray,
    """
    # list of all the verctorized images
    img_vecs = []
    for img in imgs:
        # Reshape the img to a Dx1 vector
        img_vecs.append(img.reshape((-1, 1)))

    # Stack all the images together
    img_tensor = np.hstack(img_vecs)
    # Compute the mean
    X_mean = np.mean(img_tensor, axis=1).reshape((-1, 1))

    S = np.zeros((X_mean.shape[0], X_mean.shape[0]))
    for img_vec in img_vecs:
        # Normalize X by the mean
        X_ = img_vec - X_mean
        # Get the S matrix (DxD)
        S += np.dot(X_, X_.T)

    # SVD decomposition
    eigenvectors, eigenvalues, _ = np.linalg.svd(S)

    # Sort eigenvectors and eigenvalues in ascending order
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvectors, eigenvalues, X_mean

def reconstructImages(imgs, eigenvectors, X_mean):
    """
    Compute PCA based on the input img.
    
    Returns
    -------
    rec_imgs: list[img], reconstructed imgs
    """
    rec_imgs = []
    for img in imgs:
        rec_img = X_mean + np.dot(eigenvectors, eigenvectors.T) * (img - X_mean)
        rec_imgs.append(rec_img)

    return rec_imgs


def main():
    
    # Set destination paths
    repo_path = '/Users/ss/ss_ws/face-recognition/'
    train_set_path = os.path.join(repo_path, 'data/train')
    test_set_path = os.path.join(repo_path, 'data/test')

    # test sample
    folder_path = 'data/train/3/'

    imgs = []
    for img_file in os.listdir(folder_path):
        imgs.append(cv2.imread(os.path.join(folder_path, img_file), cv2.IMREAD_GRAYSCALE))

    # Compute eigenvectors, eigenvalues, and mean
    eigenvectors, eigenvalues, mean = getPCAResults(imgs)
    print(eigenvectors.shape)
    # Get the principle vector
    v_1 = eigenvectors[:, -1]
    v_2 = eigenvectors[:, -2]
    print(v_1)
    print(v_2)

    # Display img
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()