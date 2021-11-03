import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    
    # Settings
    plot_pca_result = False
    show_rec_imgs = False

    # Set destination paths
    repo_path = '/home/ss/ss_ws/face-recognition/'
    train_set_path = os.path.join(repo_path, 'data/train')
    test_set_path = os.path.join(repo_path, 'data/test')

    # test sample
    folder_path = 'data/train/4/'

    # Load images from the train set
    img_vecs = []
    for img_file in os.listdir(folder_path):
        img_vecs.append(cv2.imread(os.path.join(folder_path, img_file), cv2.IMREAD_GRAYSCALE).reshape((1, -1)))

    # Stack image vectors together
    img_tensor = np.vstack(img_vecs)
    print(img_tensor.shape)

    # Apply PCA on 2D and 3D
    pca_2 = PCA(2)
    proj_imgs_2d = pca_2.fit_transform(img_tensor)
    pca_3 = PCA(3)
    proj_imgs_3d = pca_3.fit_transform(img_tensor)
    
    # Visualize data
    if plot_pca_result:
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


    # Apply PCA with 40, 80, and 200 Ds
    Dimensions = [40, 80, 200]
    pca_list = []
    proj_imgs_list = []

    for i in range(len(Dimensions)):
        pca_list.append(PCA(Dimensions[i]))
        proj_imgs_list.append(pca_list[i].fit_transform(img_tensor))

    print(proj_imgs_list[2].shape)
    # Reconstruct the images
    # rec_imgs_2d = pca_2.inverse_transform(proj_imgs_2d)
    # rec_imgs_3d = pca_3.inverse_transform(proj_imgs_3d)
    # print(proj_imgs_3d.shape)
    # print(rec_imgs_3d.shape)

    # cv2.imshow('reconstructed image', rec_imgs_3d[1, :].reshape((32, 32)))
    # cv2.waitKey(0)

    # Visualize reconstructed images
    if show_rec_imgs:
        for i in range(img_tensor.shape[0]):
            fig, axs = plt.subplots(1, 4)
            axs[0].title.set_text('Original')
            axs[0].imshow(img_tensor[i, :].reshape((32, 32)), cmap='gray')
            axs[1].title.set_text('D = 2')
            axs[1].imshow(rec_imgs_2d[i, :].reshape((32, 32)), cmap='gray')
            axs[2].title.set_text('D = 3')
            axs[2].imshow(rec_imgs_3d[i, :].reshape((32, 32)), cmap='gray')
            axs[2].title.set_text('D = 4')
            axs[2].imshow(rec_imgs_3d[i, :].reshape((32, 32)), cmap='gray')
            plt.show()
    
    print('Done')

if __name__ == "__main__":
    main()