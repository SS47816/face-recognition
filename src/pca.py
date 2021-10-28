import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    
    # Set destination paths
    repo_path = '/Users/ss/ss_ws/face-recognition/'
    train_set_path = os.path.join(repo_path, 'data/train')
    test_set_path = os.path.join(repo_path, 'data/test')

    # test sample
    folder_path = 'data/train/3/'

    # Load images from the train set
    img_vecs = []
    for img_file in os.listdir(folder_path):
        img_vecs.append(cv2.imread(os.path.join(folder_path, img_file), cv2.IMREAD_GRAYSCALE).reshape((-1, 1)))

    # Stack image vectors together
    img_tensor = np.hstack(img_vecs)

    # Apply PCA on 2D and 3D
    pca_2 = PCA(2)
    proj_imgs_2d = pca_2.fit_transform(img_tensor.T)
    pca_3 = PCA(3)
    proj_imgs_3d = pca_3.fit_transform(img_tensor.T)

    # Visualize data
    print('Visualizing Results... ')
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