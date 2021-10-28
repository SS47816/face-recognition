import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def getPCAResults(img_vecs, sort=True):
    """
    Compute PCA based on the input img.

    Returns
    -------
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    mean: np.ndarray,
    """
    # Stack all the image vectors together
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

def reconstructImages(img_vecs, eigenvectors, X_mean):
    """
    Compute PCA based on the input img.
    
    Returns
    -------
    rec_imgs: list[img], reconstructed imgs
    """
    rec_imgs = []
    for img_vec in img_vecs:
        rec_img = X_mean + np.dot(np.dot(eigenvectors, eigenvectors.T), (img_vec - X_mean))
        rec_imgs.append(rec_img.reshape((32, 32)))

    return rec_imgs


def main():
    
    # Set destination paths
    repo_path = '/Users/ss/ss_ws/face-recognition/'
    train_set_path = os.path.join(repo_path, 'data/train')
    test_set_path = os.path.join(repo_path, 'data/test')

    # test sample
    folder_path = 'data/train/3/'

    img_vecs = []
    for img_file in os.listdir(folder_path):
        img_vecs.append(cv2.imread(os.path.join(folder_path, img_file), cv2.IMREAD_GRAYSCALE).reshape((-1, 1)))

    # # Compute eigenvectors, eigenvalues, and mean
    # eigenvectors, eigenvalues, mean = getPCAResults(img_vecs)
    # print(eigenvectors.shape)
    # v_3 = eigenvectors[:, -3:]
    # print(v_3.shape)

    # rec_imgs = reconstructImages(img_vecs, v_3, mean)
    #     for rec_img in rec_imgs:
    #     # Display img
    #     cv2.imshow("Reconstructed image", rec_img)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Stack image vectors together
    img_tensor = np.hstack(img_vecs)
    print(img_tensor.shape)

    # Apply PCA on 2D and 3D
    pca_2 = PCA(2)
    proj_imgs_2d = pca_2.fit_transform(img_tensor.T)
    pca_3 = PCA(3)
    proj_imgs_3d = pca_3.fit_transform(img_tensor.T)
    
    print(proj_imgs_2d.shape)
    print(proj_imgs_3d.shape)

    # Visualize data
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=plt.figaspect(0.5))
    c_map = plt.cm.get_cmap('jet', 10)
    # 2D subplot
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(proj_imgs_2d[:, 0], proj_imgs_2d[:, 1], s = 15, cmap = c_map)
    ax.set_xlabel('Principle Axis 1')
    ax.set_ylabel('Principle Axis 2')
    # 3D subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(proj_imgs_3d[:, 0], proj_imgs_3d[:, 1], proj_imgs_3d[:, 2], s = 15, cmap = c_map)
    ax.set_xlabel('Principle Axis 1')
    ax.set_ylabel('Principle Axis 2')
    ax.set_zlabel('Principle Axis 3')
    plt.show()

if __name__ == "__main__":
    main()