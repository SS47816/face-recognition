import os
import random
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
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
    `MY_imgs`: `np.ndarray`, a tensor made of vertically stacked selected my_photo img vectors
    `selected_PIE_lables`: `np.ndarray`, labels of the selected PIE img vectors
    `MY_lables`: `np.ndarray`,  labels of the selected my_photo img vectors
    """
    # List all subjects in set
    set_path = os.path.join(data_path, set)
    subject_paths = os.listdir(set_path)

    # Within each subject of the PIE dataset, read all images
    PIE_imgs = []
    MY_imgs = []
    PIE_lables = []
    MY_lables = []
    idxs = []
    idx = 0
    for subject_path in subject_paths:
        folder_path = os.path.join(set_path, subject_path)
        if subject_path == '.DS_Store':
            continue
        elif subject_path == 'my_photo':
            # Load my_photo images 
            for img_file in os.listdir(folder_path):
                MY_imgs.append(cv2.imread(os.path.join(folder_path, img_file), cv2.IMREAD_GRAYSCALE).reshape((1, -1)))
                MY_lables.append(int(0))
        else:
            # Load PIE images 
            for img_file in os.listdir(folder_path):
                PIE_imgs.append(cv2.imread(os.path.join(folder_path, img_file), cv2.IMREAD_GRAYSCALE).reshape((1, -1)))
                PIE_lables.append(int(subject_path))
                idxs.append(idx)
                idx += 1
    
    if num_PIE_imgs > 0:
        # Randomly Select a given number of samples from the PIE set
        selected_idxs = random.sample(idxs, num_PIE_imgs)
        selected_PIE_imgs = [PIE_imgs[i] for i in selected_idxs]
        selected_PIE_lables = [PIE_lables[i] for i in selected_idxs]

        print('Read %d PIE images from %s and randomly sampled %d' % (len(PIE_imgs), set, len(selected_PIE_imgs)))
        print('Read %d my_photo from %s' % (len(MY_imgs), set))

        return np.vstack(selected_PIE_imgs), np.vstack(MY_imgs), np.vstack(selected_PIE_lables), np.vstack(MY_lables)
    else:
        # Return all PIE images and MY images without sampling
        print('Read %d PIE images from %s' % (len(PIE_imgs), set))
        print('Read %d my_photo from %s' % (len(MY_imgs), set))
        return np.vstack(PIE_imgs), np.vstack(MY_imgs), np.vstack(PIE_lables), np.vstack(MY_lables)


def getPCA3Results(X_train, PIE_X_train, MY_X_train, img_shape, show_plot=True) -> None:
    """
    Apply the train data to fit the PCA on 3D and plot the results in 2D and 3D

    Parameters
    ---
    `X_train`: `np.ndarray`, the train data to be used to fit the PCA algorithm
    `PIE_X_train`: `np.ndarray`, the first set of train data to be transformed by PCA
    `MY_X_train`: `np.ndarray`, the second set of train data to be transformed by PCA
    `img_shape`: `np.array`, the shape of the original images for display
    `show_plot`: `bool`, if the results should be plotted, default as `True`

    Returns
    ---
    `None`
    """
    # Apply PCA on 3D (which also included 2D)
    pca_3 = PCA(3)
    pca_3.fit(X_train)
    proj_PIE_imgs = pca_3.transform(PIE_X_train)
    proj_MY_imgs = pca_3.transform(MY_X_train)
    
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


def applyPCAs(X_train, X_test, dimensions, img_shape, show_samples=5) -> tuple:
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
    `proj_X_train_list`: `list[np.ndarray]`, list of PCA projected images on the train set
    `proj_X_test_list`: `list[np.ndarray]`, list of PCA projected images on the test set
    """
    # Apply PCA on 40, 80, 200 dimensions
    pca_list = []
    proj_X_train_list = []
    proj_X_test_list = []
    rec_imgs_list = []

    for i in range(len(dimensions)):
        pca_list.append(PCA(dimensions[i]))
        # Fit PCA on the images
        proj_X_train_list.append(pca_list[i].fit_transform(X_train))
        proj_X_test_list.append(pca_list[i].transform(X_test))
        # Reconstruct the images
        rec_imgs_list.append(pca_list[i].inverse_transform(proj_X_train_list[i]))

    # Visualize reconstructed images
    if show_samples > 0:
        print('Showing %d example results here' % show_samples)
        for i in range(show_samples):
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

    return proj_X_train_list, proj_X_test_list

def showErrorRates(x, PIE_error_rates, MY_error_rates):
    """
    Plot the error rate graph based on the given data

    Parameters
    ---
    `x`: `list[]`, list of values for the x axis
    `PIE_error_rates`: `list[np.ndarray]`, list of error rates on PIE set
    `MY_error_rates`: `list[np.ndarray]`, list of error rates on MY set
    
    Returns
    ---
    `None`
    """
    # Visualize KNN classification error rates
    fig, ax = plt.subplots()
    line1, = ax.plot(x, PIE_error_rates, marker='o', color='c', dashes=[6, 2], label='PIE test set')
    line2, = ax.plot(x, MY_error_rates, marker='*', color='r', dashes=[4, 2], label='MY test set')
    ax.set_xlabel('Image Dimensions')
    ax.set_ylabel('KNN Classification Error Rate')
    ax.legend()
    plt.show()
    return

def main():
    # Display Settings
    show_pca_result = False      # If we want to plot the PCA results
    show_num_samples = 0        # Number of example results to display after done, `0` for no output

    # Set destination paths
    data_path = '/home/ss/ss_ws/face-recognition/data'

    # Read 500 images from the train set and all from the test set
    PIE_X_train, MY_X_train, PIE_y_train, MY_y_train = readImageData(data_path, set='train', num_PIE_imgs=500)
    PIE_X_test, MY_X_test, PIE_y_test, MY_y_test = readImageData(data_path, set='test')

    # Stack all image vectors together forming train and test sets
    X_train = np.vstack((PIE_X_train, MY_X_train))
    y_train = np.vstack((PIE_y_train, MY_y_train))
    X_test = np.vstack((PIE_X_test, MY_X_test))
    y_test = np.vstack((PIE_y_test, MY_y_test))

    img_shape = np.array([np.sqrt(X_train.shape[1]), np.sqrt(X_train.shape[1])], dtype=int)

    # Apply PCA on 3D (which also included 2D) and visualize results
    getPCA3Results(X_train, PIE_X_train, MY_X_train, img_shape, show_plot=show_pca_result)

    # Apply PCA on 40, 80, 200 dimensions and show the reconstructed images
    dimensions = [40, 80, 200]
    proj_X_train_list, proj_X_test_list = applyPCAs(X_train, X_test, dimensions, img_shape, show_samples=show_num_samples)
    
    # Apply KNN Classifications on the PCA preprocessed images
    PIE_error_rates = []
    MY_error_rates = []
    # For each dimension to be tested
    for i in range(len(dimensions)):
        print('For images with %d dimensions: ' % dimensions[i])
        # Split X_test into PIE and MY datasets
        proj_PIE_X_test = proj_X_test_list[i][:-3,:]
        proj_MY_X_test = proj_X_test_list[i][-3:,:]
        # Apply KNN classifications
        KNN = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean').fit(proj_X_train_list[i], y_train.ravel())
        PIE_y_test_pred = KNN.predict(proj_PIE_X_test).reshape(-1, 1)
        MY_y_test_pred = KNN.predict(proj_MY_X_test).reshape(-1, 1)
        # Collect results
        PIE_error_rates.append((PIE_y_test_pred != PIE_y_test).sum() / PIE_y_test_pred.shape[0])
        MY_error_rates.append((MY_y_test_pred != MY_y_test).sum() / MY_y_test_pred.shape[0])

    # Apply KNN Classification on the original images
    print('For original images with %d dimensions: ' % X_train.shape[1])
    KNN = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean').fit(X_train, y_train.ravel())
    PIE_y_test_pred = KNN.predict(PIE_X_test).reshape(-1, 1)
    MY_y_test_pred = KNN.predict(MY_X_test).reshape(-1, 1)
    # Collect results
    PIE_error_rates.append((PIE_y_test_pred != PIE_y_test).sum() / PIE_y_test_pred.shape[0])
    MY_error_rates.append((MY_y_test_pred != MY_y_test).sum() / MY_y_test_pred.shape[0])
    
    # Visualize KNN classification error rates
    dimensions.append(PIE_X_test.shape[1])
    showErrorRates(dimensions, PIE_error_rates, MY_error_rates)

    print('Finished PCA Processing')
    return

if __name__ == "__main__":
    main()