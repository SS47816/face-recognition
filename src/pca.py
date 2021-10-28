import os
import math
import numpy as np
import cv2
import torch

def getPCAResults(imgs, sort=True):
    """Compute PCA based on the input img.
    Return: 
        float: mean
        float: 
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

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvectors, eigenvalues, X_mean


def main():
    folder_path = 'data/PIE/1/'
    imgs = []
    for img_file in os.listdir(folder_path):
        imgs.append(cv2.imread(os.path.join(folder_path, img_file), cv2.IMREAD_GRAYSCALE))

    # Compute eigenvectors, eigenvalues, and mean
    eigenvectors, eigenvalues, mean = getPCAResults(imgs)
    print(eigenvectors.shape)
    # Get the principle vector
    v_1 = eigenvectors[:, -1]
    print(v_1.shape)

    # Display img
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()