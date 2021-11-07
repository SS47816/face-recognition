import os
import random
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

def getPCA3Results(X_train, PIE_X_train, MY_X_train, show_plot=True) -> None:
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
        img_shape = np.array([np.sqrt(X_train.shape[1]), np.sqrt(X_train.shape[1])], dtype=int)
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

def applyPCAs(X_train, X_test, dimensions, show_samples=5) -> tuple:
    """
    Apply the train data to fit a series of PCAs with different dimensions and show the reconstructed images

    Parameters
    ---
    `X_train`: `np.ndarray`, the train data to be used to fit the PCA algorithm
    `dimensions`: `list[int]`, list of PCA dimensions to be tested
    `X_test`: `np.ndarray`, the test data to be transformed by the PCA algorithm
    `show_samples`: `int`, the number of example results to display after done, `0` for no output, default as `5`
    
    Returns
    ---
    `proj_X_train_list`: `list[np.ndarray]`, list of train set images after PCA dimensionality reduction
    `proj_X_test_list`: `list[np.ndarray]`, list of test set images after PCA dimensionality reduction
    """
    # Apply PCA on a list of dimensions
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
    img_shape = np.array([np.sqrt(X_train.shape[1]), np.sqrt(X_train.shape[1])], dtype=int)
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

def KNNClassification(X_train, y_train, PIE_X_test, PIE_y_test, MY_X_test, MY_y_test):
    """
    Apply KNN classification on the train data and predict on two separate sets

    Parameters
    ---
    `X_train`: `np.ndarray`, the train data X to be used to fit the KNN classifier
    `y_train`: `np.ndarray`, the train data y to be used to fit the KNN classifier
    `PIE_X_test`: `np.ndarray`, the first test set to be used by the KNN classifier to predict
    `PIE_y_test`: `np.ndarray`, the first test set's y_label to be used to compute error rate
    `MY_X_test`: `np.ndarray`, the second test set to be used by the KNN classifier to predict
    `MY_y_test`: `np.ndarray`, the second test set's y_label to be used to compute error rate
    
    Returns
    ---
    `PIE_error_rate`: `float`, list of PCA projected images on the train set
    `MY_error_rate`: `float`, list of PCA projected images on the test set
    """
    # Apply KNN classifications
    KNN = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean').fit(X_train, y_train.ravel())
    PIE_y_test_pred = KNN.predict(PIE_X_test).reshape(-1, 1)
    MY_y_test_pred = KNN.predict(MY_X_test).reshape(-1, 1)
    # Collect results
    PIE_error_rate = (PIE_y_test_pred != PIE_y_test).sum() / PIE_y_test_pred.shape[0]
    MY_error_rate = (MY_y_test_pred != MY_y_test).sum() / MY_y_test_pred.shape[0]

    return PIE_error_rate, MY_error_rate

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


def applyLDAs(X_train, y_train, X_test, dimensions) -> tuple:
    """
    Apply the train data to fit a series of LDAs with different dimensions and show the reconstructed images

    Parameters
    ---
    `X_train`: `np.ndarray`, the train data to be used to fit the LDA algorithm
    `y_train`: `np.ndarray`, the train data label to be used to fit the LDA algorithm
    `X_test`: `np.ndarray`, the test data to be transformed by the LDA algorithm
    `dimensions`: `list[int]`, list of LDA dimensions to be tested
    
    Returns
    ---
    `proj_X_train_list`: `list[np.ndarray]`, list of train set images after LDA dimensionality reduction
    `proj_X_test_list`: `list[np.ndarray]`, list of test set images after LDA dimensionality reduction
    """
    # Apply LDA on a list of dimensions
    lda_list = []
    proj_X_train_list = []
    proj_X_test_list = []

    for i in range(len(dimensions)):
        lda_list.append(LinearDiscriminantAnalysis(n_components=dimensions[i]))
        # Fit LDA on the images
        lda_list[i].fit(X_train, y_train.ravel())
        proj_X_train_list.append(lda_list[i].transform(X_train))
        proj_X_test_list.append(lda_list[i].transform(X_test))

    return proj_X_train_list, proj_X_test_list


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

    # Apply PCA on 3D (which also included 2D) and visualize results
    getPCA3Results(X_train, PIE_X_train, MY_X_train, show_plot=show_pca_result)

    # Apply PCA on 40, 80, 200 dimensions and show the reconstructed images
    dimensions = [40, 80, 200]
    proj_X_train_list, proj_X_test_list = applyPCAs(X_train, X_test, dimensions, show_samples=show_num_samples)
    
    # Apply KNN Classifications on the PCA preprocessed images
    PIE_error_rates = []
    MY_error_rates = []
    # For each dimension to be tested
    for i in range(len(proj_X_train_list)):
        print('For images with %d dimensions: ' % dimensions[i])
        # Split X_test into PIE and MY datasets
        proj_PIE_X_test = proj_X_test_list[i][:-3,:]
        proj_MY_X_test = proj_X_test_list[i][-3:,:]
        print(proj_PIE_X_test.shape)
        print(PIE_y_test.shape)
        # Apply KNN classifications
        PIE_error_rate, MY_error_rate = KNNClassification(proj_X_train_list[i], y_train, proj_PIE_X_test, PIE_y_test, proj_MY_X_test, MY_y_test)
        # Collect results
        PIE_error_rates.append(PIE_error_rate)
        MY_error_rates.append(MY_error_rate)

    # Apply KNN Classification on the original images
    print('For original images with %d dimensions: ' % X_train.shape[1])
    PIE_error_rate, MY_error_rate = KNNClassification(X_train, y_train, PIE_X_test, PIE_y_test, MY_X_test, MY_y_test)
    # Collect results
    PIE_error_rates.append(PIE_error_rate)
    MY_error_rates.append(MY_error_rate)
    
    # Visualize KNN classification error rates
    dimensions.append(PIE_X_test.shape[1])
    showErrorRates(dimensions, PIE_error_rates, MY_error_rates)

    print('Finished PCA Processing')

    # Apply LDA to reduce the dimensionalities of the original images
    dimensions = [2, 3, 9]
    proj_X_train_list, proj_X_test_list = applyLDAs(X_train, y_train, X_test, dimensions)
    
    # Read the whole train and test set

    return

if __name__ == "__main__":
    main()