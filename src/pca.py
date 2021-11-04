import os
import random
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def readImageData(data_path, set='train', num_PIE_imgs=-1) -> tuple:
    """
    Read the PIE dataset and my_photo dataset

    Parameters
    ---
    `data_path`: `string`, path to the data folder
    `set`: `string`, can be either `train` or `test`
    `num_PIE_imgs`: `int`, number of PIE images to sample

    Returns
    ---
    `selected_PIE_imgs`: `np.ndarray`, a tensor made of vertically stacked selected PIE img vectors
    `my_imgs`: `np.ndarray`, a tensor made of vertically stacked selected my_photo img vectors
    """
    # List all subjects in set
    set_path = os.path.join(data_path, set)
    subject_paths = os.listdir(set_path)

    # Within each subject of the PIE dataset, read all images
    PIE_imgs = []
    my_imgs = []
    for subject_path in subject_paths:
        folder_path = os.path.join(set_path, subject_path)
        if subject_path == '.DS_Store':
            continue
        elif subject_path == 'my_photo':
            # Load my_photo images 
            for img_file in os.listdir(folder_path):
                my_imgs.append(cv2.imread(os.path.join(folder_path, img_file), cv2.IMREAD_GRAYSCALE).reshape((1, -1)))
        else:
            # Load PIE images 
            for img_file in os.listdir(folder_path):
                PIE_imgs.append(cv2.imread(os.path.join(folder_path, img_file), cv2.IMREAD_GRAYSCALE).reshape((1, -1)))
    
    # Randomly Select a given number of samples from the PIE set
    selected_PIE_imgs = random.sample(PIE_imgs, num_PIE_imgs)

    print('Read %d PIE images from %s' % (len(selected_PIE_imgs), set))
    print('Read %d my_photo from %s' % (len(my_imgs), set))

    return np.vstack(selected_PIE_imgs), np.vstack(my_imgs)


def getPCA3Results(X_train, PIE_train, MY_train, img_shape, show_plot=True) -> None:
    """
    Apply the train data to fit the PCA on 3D and plot the results in 2D and 3D

    Parameters
    ---
    `X_train`: `np.ndarray`, the train data to be used to fit the PCA algorithm
    `PIE_train`: `np.ndarray`, the first set of train data to be transformed by PCA
    `MY_train`: `np.ndarray`, the second set of train data to be transformed by PCA
    `img_shape`: `np.array`, the shape of the original images for display
    `show_plot`: `bool`, if the results should be plotted, default as `True`

    Returns
    ---
    `None`
    """
    # Apply PCA on 3D (which also included 2D)
    pca_3 = PCA(3)
    pca_3.fit(X_train)
    proj_PIE_imgs = pca_3.transform(PIE_train)
    proj_MY_imgs = pca_3.transform(MY_train)
    
    # Visualize results
    if show_plot:
        print('Visualizing Results... ')
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=plt.figaspect(0.5))
        # 2D Plot
        ax = fig.add_subplot(1, 2, 1)
        ax.scatter(proj_PIE_imgs[:, 0], proj_PIE_imgs[:, 1], s = 10, c = 'c')
        ax.scatter(proj_MY_imgs[:, 0], proj_MY_imgs[:, 1], s = 15, c = 'r')
        ax.set_xlabel('Principle Axis 1')
        ax.set_ylabel('Principle Axis 2')
        # 3D Plot
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(proj_PIE_imgs[:, 0], proj_PIE_imgs[:, 1], proj_PIE_imgs[:, 2], s = 10, c = 'c')
        ax.scatter(proj_MY_imgs[:, 0], proj_MY_imgs[:, 1], proj_MY_imgs[:, 2], s = 15, c = 'r')
        ax.set_xlabel('Principle Axis 1')
        ax.set_ylabel('Principle Axis 2')
        ax.set_zlabel('Principle Axis 3')
        plt.show()

        # Plot the mean face and eigen faces
        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(1, 4, 1, xticks=[], yticks=[])
        ax.imshow(pca_3.mean_.reshape(img_shape), cmap='gray')
        for i in range(3):
            ax = fig.add_subplot(1, 4, i + 2, xticks=[], yticks=[])
            ax.imshow(pca_3.components_[i].reshape(img_shape), cmap='gray')
        plt.show()
    return


def reconstructImgsPCAs(X_train, dimensions, img_shape, show_samples=5) -> None:
    """
    Apply the train data to fit a series of PCAs with different dimensions and show the reconstructed images

    Parameters
    ---
    `X_train`: `np.ndarray`, the train data to be used to fit the PCA algorithm
    `dimensions`: `list[int]`, list of PCA dimensions to be tested
    `img_shape`: `np.array`, the shape of the original images for display
    `show_samples`: `int`, the number of example results to display after done, `0` for no output, default as `5`
    
    Returns
    ---
    `None`
    """
    # Apply PCA on 40, 80, 200 dimensions
    pca_list = []
    proj_imgs_list = []
    rec_imgs_list = []

    for i in range(len(dimensions)):
        pca_list.append(PCA(dimensions[i]))
        # Fit PCA on the images
        proj_imgs_list.append(pca_list[i].fit_transform(X_train))
        # Reconstruct the images
        rec_imgs_list.append(pca_list[i].inverse_transform(proj_imgs_list[i]))

    # Visualize reconstructed images
    if show_samples > 0:
        print('Showing %d example results here' % show_samples)
        for i in range(X_train.shape[0]):
            if (i < show_samples):
                # Plot the original image and the reconstructed faces
                fig = plt.figure(figsize=(16, 6))
                ax = fig.add_subplot(1, 4, 1, xticks=[], yticks=[])
                ax.title.set_text('Original')
                ax.imshow(X_train[i, :].reshape(img_shape), cmap='gray')
                for j in range(3):
                    ax = fig.add_subplot(1, 4, j + 2, xticks=[], yticks=[])
                    ax.title.set_text('D = %d' %dimensions[j])
                    ax.imshow(rec_imgs_list[j][i, :].reshape(img_shape), cmap='gray')
                plt.show()
            else:
                break
    return


def main():
    # Display Settings
    show_pca_result = True      # If we want to plot the PCA results
    show_num_samples = 3        # Number of example results to display after done, `0` for no output

    # Set destination paths
    data_path = '/home/ss/ss_ws/face-recognition/data'

    # Read 500 images from the train set
    PIE_train, MY_train = readImageData(data_path, set='train', num_PIE_imgs=500)

    # Stack all image vectors together forming X_train set
    X_train = np.vstack((PIE_train, MY_train))
    img_shape = np.array([np.sqrt(X_train.shape[1]), np.sqrt(X_train.shape[1])], dtype=int)
    print(X_train.shape)

    # Apply PCA on 3D (which also included 2D) and visualize results
    getPCA3Results(X_train, PIE_train, MY_train, img_shape, show_plot=show_pca_result)

    # Apply PCA on 40, 80, 200 dimensions and show the reconstructed images
    dimensions = [40, 80, 200]
    reconstructImgsPCAs(X_train, dimensions, img_shape, show_samples=show_num_samples)

    print('Finished PCA Processing')
    return

if __name__ == "__main__":
    main()