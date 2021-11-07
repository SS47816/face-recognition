import os
import random
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

class FaceDataset:
    """
    Customized data structure for storing all face images in a organized way

    Arrtributes
    -----------
    `X_PIE` (`np.ndarray`): the PIE training set images
    `y_PIE` (`np.ndarray`): the PIE training set labels
    `X_MY` (`np.ndarray`): my_photo training set images
    `y_MY` (`np.ndarray`): my_photo training set labels
    `X` (`np.ndarray`): the whole training set images
    `y` (`np.ndarray`): the whole training set labels
    `name` (`str`): the name of the set, can be either `train` or `test`
    """
    def __init__(self, X_PIE: np.ndarray, y_PIE: np.ndarray, X_MY: np.ndarray, y_MY: np.ndarray, name: str=''):
        """
        Initialize the dataset by combining PIE and MY images

        Arrtributes
        -----------
        `X_PIE` (`np.ndarray`): the PIE training set images
        `y_PIE` (`np.ndarray`): the PIE training set labels
        `X_MY` (`np.ndarray`): my_photo training set images
        `y_MY` (`np.ndarray`): my_photo training set labels
        `X` (`np.ndarray`): the whole training set images
        `y` (`np.ndarray`): the whole training set labels
        `name` (`str`): the name of the set, can be either `train` or `test`
        """
        self.name = name
        self.X_PIE = X_PIE
        self.y_PIE = y_PIE
        self.X_MY = X_MY
        self.y_MY = y_MY
        # Stack all image vectors together forming train and test sets
        self.X = np.vstack((self.X_PIE, self.X_MY))
        self.y = np.vstack((self.y_PIE, self.y_MY))

    @classmethod
    def split(cls, X: np.ndarray, y: np.ndarray, split_idx: int, name :str=''):
        """
        Altaernative constructor, create the dataset by spliting the a whole set

        Arrtributes
        -----------
        `X` (`np.ndarray`): the PIE training set images
        `y` (`np.ndarray`): the PIE training set labels
        `split_idx` (`int`): the index used to split the PIE set and MY set 
        `name` (`str`): the name of the set, can be either `train` or `test`
        """
        X_PIE = X[:split_idx,:]
        y_PIE = y[:split_idx,:]
        X_MY = X[split_idx:,:]
        y_MY = y[split_idx:,:]
        return cls(X_PIE, y_PIE, X_MY, y_MY, name)

def readImageData(data_path: str, set: str='train', num_PIE_imgs: int=-1) -> FaceDataset:
    """
    Read the PIE dataset and my_photo dataset

    Parameters
    ----------
    `data_path` (`str`): path to the data folder
    `set` (`str`): can be either `train` or `test`
    `num_PIE_imgs` (`int`): number of PIE images to sample

    Returns
    -------
    `FaceDataset`: the organized dataset ready to be used
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

        return FaceDataset(np.vstack(selected_PIE_imgs), np.vstack(selected_PIE_lables), np.vstack(MY_imgs), np.vstack(MY_lables), name=set)
    else:
        # Return all PIE images and MY images without sampling
        print('Read %d PIE images from %s' % (len(PIE_imgs), set))
        print('Read %d my_photo from %s' % (len(MY_imgs), set))
        return FaceDataset(np.vstack(PIE_imgs), np.vstack(PIE_lables),np.vstack(MY_imgs), np.vstack(MY_lables), name=set)

def getPCA3Results(train: FaceDataset, show_plot: bool=True) -> None:
    """
    Apply the train data to fit the PCA on 3D and plot the results in 2D and 3D

    Parameters
    ----------
    `X_train` (`np.ndarray`): the train data to be used to fit the PCA algorithm
    `PIE_X_train` (`np.ndarray`): the first set of train data to be transformed by PCA
    `MY_X_train` (`np.ndarray`): the second set of train data to be transformed by PCA
    `img_shape` (`np.array`): the shape of the original images for display
    `show_plot` (`bool`): if the results should be plotted, default as `True`

    Returns
    -------
    `None`
    """
    # Apply PCA on 3D (which also included 2D)
    pca_3 = PCA(3)
    pca_3.fit(train.X)
    proj_PIE_imgs = pca_3.transform(train.X_PIE)
    proj_MY_imgs = pca_3.transform(train.X_MY)
    
    # Visualize results
    if show_plot:
        print('Visualizing Results... ')
        img_shape = np.array([np.sqrt(train.X.shape[1]), np.sqrt(train.X.shape[1])], dtype=int)
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

def applyPCAs(dimensions, X_train, X_test, show_samples=5) -> tuple:
    """
    Apply the train data to fit a series of PCAs with different dimensions and show the reconstructed images

    Parameters
    ----------
    `X_train` (`np.ndarray`): the train data to be used to fit the PCA algorithm
    `dimensions` (`list[int]`): list of PCA dimensions to be tested
    `X_test` (`np.ndarray`): the test data to be transformed by the PCA algorithm
    `show_samples` (`int`): the number of example results to display after done, `0` for no output, default as `5`
    
    Returns
    -------
    `proj_X_train_list` (`list[np.ndarray]`): list of train set images after PCA dimensionality reduction
    `proj_X_test_list` (`list[np.ndarray]`): list of test set images after PCA dimensionality reduction
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

def KNNClassification(train_set: FaceDataset, test_set: FaceDataset) -> np.ndarray:
    """
    Apply KNN classification on the train data and predict on two separate sets

    Parameters
    ----------
    `train_set` (`FaceDataset`): the train data X to be used to train the KNN classifier
    `test_set` (`FaceDataset`): the test data to be tested with the KNN classifier
    
    Returns
    -------
    `PIE_error_rate` (`float`): list of classification error rates on the train set
    `MY_error_rate` (`float`): list of classification error rates on the test set
    """
    # Apply KNN classifications
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean').fit(train_set.X, train_set.y.ravel())
    print(test_set.X_PIE.shape)
    print(test_set.X_MY.shape)
    PIE_y_test_pred = knn.predict(test_set.X_PIE).reshape(-1, 1)
    MY_y_test_pred = knn.predict(test_set.X_MY).reshape(-1, 1)
    # Collect results
    print(PIE_y_test_pred.shape)
    print(test_set.y_PIE.shape)
    error_rate = np.zeros((2,1))
    error_rate[0, 0] = (PIE_y_test_pred != test_set.y_PIE).sum() / PIE_y_test_pred.shape[0]
    error_rate[1, 0] = (MY_y_test_pred != test_set.y_MY).sum() / MY_y_test_pred.shape[0]

    return error_rate

def KNNClassifications(dimensions, proj_X_train_list, proj_X_test_list, y_train, y_test) -> np.ndarray:
    """
    Apply KNN Classifications on the (preprocessed) images
    """
    error_rates = np.empty(shape=(2,0), dtype=np.float)
    # For each dimension to be tested
    for i in range(len(proj_X_train_list)):
        print('For images with %d dimensions: ' % dimensions[i])
        # Construct datasets for classification
        print(proj_X_train_list[i].shape)
        print(proj_X_test_list[i].shape)
        proj_train_set = FaceDataset.split(proj_X_train_list[i], y_train, -3, name='train')
        proj_test_set = FaceDataset.split(proj_X_test_list[i], y_test, -3, name='test')
        # Apply KNN classifications
        error_rates = np.append(error_rates, KNNClassification(proj_train_set, proj_test_set), axis=1)

    return error_rates

def showErrorRates(x, pca_error_rates):
    """
    Plot the error rate graph based on the given data

    Parameters
    ----------
    `x` (`list[]`): list of values for the x axis
    `PIE_error_rates` (`list[np.ndarray]`): list of error rates on PIE set
    `MY_error_rates` (`list[np.ndarray]`): list of error rates on MY set
    
    Returns
    -------
    `None`
    """
    # Visualize KNN classification error rates
    fig, ax = plt.subplots()
    line1, = ax.plot(x, pca_error_rates[0], marker='o', color='c', dashes=[6, 2], label='PIE test set')
    line2, = ax.plot(x, pca_error_rates[1], marker='*', color='r', dashes=[4, 2], label='MY test set')
    ax.set_xlabel('Image Dimensions')
    ax.set_ylabel('KNN Classification Error Rate')
    ax.legend()
    plt.show()
    return

def applyLDAs(dimensions, X_train, y_train, X_test) -> tuple:
    """
    Apply the train data to fit a series of LDAs with different dimensions and show the reconstructed images

    Parameters
    ----------
    `X_train` (`np.ndarray`): the train data to be used to fit the LDA algorithm
    `y_train` (`np.ndarray`): the train data label to be used to fit the LDA algorithm
    `X_test` (`np.ndarray`): the test data to be transformed by the LDA algorithm
    `dimensions` (`list[int]`): list of LDA dimensions to be tested
    
    Returns
    -------
    `proj_X_train_list` (`list[np.ndarray]`): list of train set images after LDA dimensionality reduction
    `proj_X_test_list` (`list[np.ndarray]`): list of test set images after LDA dimensionality reduction
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
    train_set = readImageData(data_path, set='train', num_PIE_imgs=500)
    test_set = readImageData(data_path, set='test')

    # Apply PCA on 3D (which also included 2D) and visualize results
    getPCA3Results(train_set, show_plot=show_pca_result)

    # Apply PCA on 40, 80, 200 dimensions and show the reconstructed images
    dimensions = [40, 80, 200]
    proj_X_train_list, proj_X_test_list = applyPCAs(dimensions, train_set.X, test_set.X, show_samples=show_num_samples)
    
    # Apply KNN Classifications on the PCA preprocessed image lists
    pca_error_rates = KNNClassifications(dimensions, proj_X_train_list, proj_X_test_list, train_set.y, test_set.y)

    # Apply KNN Classification on the original images (as a baseline for comparison later)
    print('For original images with %d dimensions: ' % train_set.X.shape[1])
    pca_error_rates = np.append(pca_error_rates, KNNClassification(train_set, test_set), axis=1)
    
    # Visualize KNN classification error rates
    dimensions.append(train_set.X.shape[1])
    showErrorRates(dimensions, pca_error_rates)
    print('Finished PCA Processing')

    # Apply LDA to reduce the dimensionalities of the original images
    dimensions = [2, 3, 9]
    proj_X_train_list, proj_X_test_list = applyLDAs(dimensions, train_set.X, train_set.y, test_set.X)
    
    # Read the whole train and test set

    return

if __name__ == "__main__":
    main()