import os
import pathlib
import random
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
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
    Read the PIE dataset and my_photo dataset and return as a FaceDataset object

    Parameters
    ----------
    `data_path` (`str`): path to the data folder
    `set` (`str`): can be either `train` or `test`
    `num_PIE_imgs` (`int`): number of PIE images to sample, default `-1` for read all images

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
                if img_file == '.DS_Store':
                    continue
                MY_imgs.append(cv2.imread(os.path.join(folder_path, img_file), cv2.IMREAD_GRAYSCALE).reshape((1, -1)))
                MY_lables.append(int(0))
        else:
            # Load PIE images 
            for img_file in os.listdir(folder_path):
                if img_file == '.DS_Store':
                    continue
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

def plotProjectedData(proj_PIE_3d: np.ndarray, proj_MY_3d: np.ndarray) -> None:
    """
    Plot the projected data onto 2D and 3D scatter plots respectively

    Parameters
    ----------
    `proj_PIE_3d` (`np.ndarray`): projected data 1 to be plotted
    `proj_MY_3d` (`np.ndarray`): projected data 2 to be plotted

    Returns
    -------
    `None`
    """
    print('Visualizing Results... ')
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # 2D Plot
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(proj_PIE_3d[:, 0], proj_PIE_3d[:, 1], s = 10, c = 'c')
    ax.scatter(proj_MY_3d[:, 0], proj_MY_3d[:, 1], s = 15, c = 'r')
    ax.set_xlabel('Principle Axis 1')
    ax.set_ylabel('Principle Axis 2')
    # 3D Plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(proj_PIE_3d[:, 0], proj_PIE_3d[:, 1], proj_PIE_3d[:, 2], s = 10, c = 'c')
    ax.scatter(proj_MY_3d[:, 0], proj_MY_3d[:, 1], proj_MY_3d[:, 2], s = 15, c = 'r')
    ax.set_xlabel('Principle Axis 1')
    ax.set_ylabel('Principle Axis 2')
    ax.set_zlabel('Principle Axis 3')
    plt.show()

def plotPCA3DResults(train: FaceDataset, show_plot: bool=True) -> None:
    """
    Apply the train data to fit the PCA on 3D and plot the results in 2D and 3D

    Parameters
    ----------
    `train` (`FaceDataset`): the training dataset for the PCA to reduce dimensionality
    `show_plot` (`bool`): if the results should be plotted, default as `True`

    Returns
    -------
    `None`
    """
    # Apply PCA on 3D (which also included 2D)
    pca_3 = PCA(3)
    pca_3.fit(train.X)
    proj_PIE_3d = pca_3.transform(train.X_PIE)
    proj_MY_3d = pca_3.transform(train.X_MY)
    
    # Visualize results
    if show_plot:
        # Plot the projected data onto 2D and 3D scatter plots
        plotProjectedData(proj_PIE_3d, proj_MY_3d)
        # Plot the mean face and eigen faces
        img_shape = np.array([np.sqrt(train.X.shape[1]), np.sqrt(train.X.shape[1])], dtype=int)
        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(1, 4, 1, xticks=[], yticks=[])
        ax.imshow(pca_3.mean_.reshape(img_shape), cmap='gray')
        for i in range(3):
            ax = fig.add_subplot(1, 4, i + 2, xticks=[], yticks=[])
            ax.imshow(pca_3.components_[i].reshape(img_shape), cmap='gray')
        plt.show()
        
    return

def applyPCAs(dims: list, train: FaceDataset, test: FaceDataset, show_samples: int=5) -> tuple:
    """
    Apply the train data to fit a series of PCAs with different dimensions and show the reconstructed images

    Parameters
    ----------
    `dimensions` (`list[int]`): list of PCA dimensions to be tested
    `train` (`FaceDataset`): the train data to be used to fit the PCA algorithm
    `test` (`FaceDataset`): the test data to be transformed by the PCA algorithm
    `show_samples` (`int`): the number of example results to display after done, `0` for no output, default as `5`
    
    Returns
    -------
    `list[FaceDataset]`: list of train set images after PCA dimensionality reduction
    `list[FaceDataset]`: list of test set images after PCA dimensionality reduction
    """
    # Apply PCA on a list of dimensions
    pca_list = []
    rec_imgs_list = []
    proj_train_list = []
    proj_test_list = []
    for i in range(len(dims)):
        pca_list.append(PCA(dims[i]))
        # Fit PCA on the images
        proj_train_X = pca_list[i].fit_transform(train.X)
        proj_test_X = pca_list[i].transform(test.X)
        proj_train_list.append(FaceDataset.split(proj_train_X, train.y, -7, name='train'))
        proj_test_list.append(FaceDataset.split(proj_test_X, train.y, -3, name='test'))
        # Reconstruct the images
        rec_imgs_list.append(pca_list[i].inverse_transform(proj_train_X))

    # Visualize reconstructed images
    img_shape = np.array([np.sqrt(train.X.shape[1]), np.sqrt(train.X.shape[1])], dtype=int)
    if show_samples > 0:
        print('Showing %d example results here' % show_samples)
        for i in range(show_samples):
            # Plot the original image and the reconstructed faces
            fig = plt.figure(figsize=(16, 6))
            ax = fig.add_subplot(1, 4, 1, xticks=[], yticks=[])
            ax.title.set_text('Original')
            ax.imshow(train.X[i, :].reshape(img_shape), cmap='gray')
            for j in range(3):
                ax = fig.add_subplot(1, 4, j + 2, xticks=[], yticks=[])
                ax.title.set_text('D = %d' %dims[j])
                ax.imshow(rec_imgs_list[j][i, :].reshape(img_shape), cmap='gray')
            plt.show()

    return proj_train_list, proj_test_list

def KNNClassification(train_set: FaceDataset, test_set: FaceDataset) -> np.ndarray:
    """
    Apply KNN classification on the training set data and predict on the test set

    Parameters
    ----------
    `train_set` (`FaceDataset`): the train data X to be used to train the KNN classifier
    `test_set` (`FaceDataset`): the test data to be tested with the KNN classifier
    
    Returns
    -------
    `np.ndarray`: classification error rates with a shape of `[2, 1]`
    """
    # Apply KNN classifications
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean').fit(train_set.X, train_set.y.ravel())
    PIE_y_test_pred = knn.predict(test_set.X_PIE).reshape(-1, 1)
    MY_y_test_pred = knn.predict(test_set.X_MY).reshape(-1, 1)
    # Collect results
    error_rate = np.zeros((2,1))
    error_rate[0, 0] = (PIE_y_test_pred != test_set.y_PIE).sum() / PIE_y_test_pred.shape[0]
    error_rate[1, 0] = (MY_y_test_pred != test_set.y_MY).sum() / MY_y_test_pred.shape[0]

    return error_rate

def KNNClassifications(train_list: list, test_list: list) -> np.ndarray:
    """
    Apply KNN Classifications on the (preprocessed) images

    Parameters
    ----------
    `train_list` (`list[FaceDataset]`): the list of train datasets to be used to train the KNN classifier
    `test_list` (`list[FaceDataset]`): the list of test datasets to be tested with the KNN classifier
    
    Returns
    -------
    `np.ndarray`: classification error rates with a shape of `[2, len(proj_X_train_list)]`
    """
    error_rates = np.empty(shape=(2,0), dtype=float)
    # For each dimension to be tested
    for i in range(len(train_list)):
        # Apply KNN classifications and store results
        error_rates = np.append(error_rates, KNNClassification(train_list[i], train_list[i]), axis=1)

    return error_rates

def showErrorRates(x: list, error_rates: list) -> None:
    """
    Plot the error rate graph based on the given data

    Parameters
    ----------
    `x` (`list[]`): list of values used for x axis
    `error_rates` (`list[np.ndarray]`): list of error rates on PIE and MY test sets (y axis)
    
    Returns
    -------
    `None`
    """
    # Visualize KNN classification error rates
    print(x)
    print(error_rates)
    fig, ax = plt.subplots()
    line1, = ax.plot(x, error_rates[0], marker='o', color='c', dashes=[6, 2], label='PIE test set')
    line2, = ax.plot(x, error_rates[1], marker='*', color='r', dashes=[4, 2], label='MY test set')
    ax.set_xlabel('Image Dimensions')
    ax.set_ylabel('KNN Classification Error Rate')
    ax.legend()
    plt.show()
    return

def applyLDAs(dims: list, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Apply the train data to fit a series of LDAs with different dimensions and show the reconstructed images

    Parameters
    ----------
    `dimensions` (`list[int]`): list of LDA dimensions to be tested
    `X_train` (`np.ndarray`): the train data to be used to fit the LDA algorithm
    `y_train` (`np.ndarray`): the train data label to be used to fit the LDA algorithm
    `X_test` (`np.ndarray`): the test data to be transformed by the LDA algorithm
    
    Returns
    -------
    `list[np.ndarray]`: list of train set images after LDA dimensionality reduction
    `list[np.ndarray]`: list of test set images after LDA dimensionality reduction
    """
    # Apply LDA on a list of dimensions
    lda_list = []
    proj_X_train_list = []
    proj_X_test_list = []

    for i in range(len(dims)):
        lda_list.append(LinearDiscriminantAnalysis(n_components=dims[i]))
        # Fit LDA on the images
        lda_list[i].fit(X_train, y_train.ravel())
        proj_X_train_list.append(lda_list[i].transform(X_train))
        proj_X_test_list.append(lda_list[i].transform(X_test))

    return proj_X_train_list, proj_X_test_list


def main():
    # Display Settings
    show_pca_result = True      # If we want to plot the PCA results
    show_num_samples = 0        # Number of example results to display after done, `0` for no output
    show_lda_result = True      # If we want to plot the PCA results

    # Set destination paths
    repo_path = pathlib.Path(__file__).parent.parent.resolve()
    data_path = os.path.join(repo_path, 'data')
    print(data_path)
    # data_path = '/home/ss/ss_ws/face-recognition/data'

    # Read 500 images from the train set and all from the test set
    train_set = readImageData(data_path, set='train', num_PIE_imgs=500)
    test_set = readImageData(data_path, set='test')

    # Apply PCA on 3D (which also included 2D) and visualize results
    plotPCA3DResults(train_set, show_plot=show_pca_result)

    # Apply PCA on 40, 80, 200 dimensions and show the reconstructed images
    pca_dims = [40, 80, 200]
    proj_train_list, proj_test_list = applyPCAs(pca_dims, train_set, test_set, show_samples=show_num_samples)
    
    # Apply KNN Classifications on the PCA preprocessed image lists
    pca_error_rates = KNNClassifications(proj_train_list, proj_test_list)

    # Apply KNN Classification on the original images (as a baseline for comparison later)
    pca_dims.append(train_set.X.shape[1])
    baseline_error_rate = KNNClassification(train_set, test_set)
    pca_error_rates = np.append(pca_error_rates, baseline_error_rate, axis=1)
    
    # Visualize KNN classification error rates
    showErrorRates(pca_dims, pca_error_rates)
    print('Finished Task 1: PCA')

    # Apply LDA to reduce the dimensionalities of the original images
    lda_dims = [2, 3, 9]
    proj_X_train_list, proj_X_test_list = applyLDAs(lda_dims, train_set.X, train_set.y, test_set.X)
    # Visualize results
    if show_lda_result:
        # Plot the projected data onto 2D and 3D scatter plots
        plotProjectedData(proj_X_train_list[1][:-7,:], proj_X_train_list[1][-7:,:])

    lda_error_rates = KNNClassifications(proj_X_train_list, proj_X_test_list, train_set.y, test_set.y)
    showErrorRates(lda_dims, lda_error_rates)
    print('Finished Task 2: LDA')
    
    # Read the whole train and test set (unlike the first time which we only sampled 500)
    train_set = readImageData(data_path, set='train')
    test_set = readImageData(data_path, set='test')
    # Apply PCA preprocessing on all the images
    pca_dims = [80, 200]
    proj_X_train_list, proj_X_test_list = applyPCAs(pca_dims, train_set.X, test_set.X, show_samples=show_num_samples)

    return

if __name__ == "__main__":
    main()