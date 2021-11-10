import os
import pathlib
import random
import numpy as np
import pandas as pd
import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
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
        `y_PIE` (`np.array`): the PIE training set labels
        `X_MY` (`np.ndarray`): my_photo training set images
        `y_MY` (`np.array`): my_photo training set labels
        `X` (`np.ndarray`): the whole training set images
        `y` (`np.array`): the whole training set labels
        `name` (`str`): the name of the set, can be either `train` or `test`
        """
        self.name = name
        self.X_PIE = X_PIE
        self.y_PIE = y_PIE.ravel()
        self.X_MY = X_MY
        self.y_MY = y_MY.ravel()
        # Stack all image vectors together forming train and test sets
        self.X = np.vstack((self.X_PIE, self.X_MY))
        self.y = np.concatenate((self.y_PIE, self.y_MY))

    @classmethod
    def split(cls, X: np.ndarray, y: np.ndarray, name :str='train'):
        """
        Altaernative constructor, create the dataset by spliting the a whole set

        Arrtributes
        -----------
        `X` (`np.ndarray`): the PIE training set images
        `y` (`np.ndarray`): the PIE training set labels
        `split_idx` (`int`): the index used to split the PIE set and MY set 
        `name` (`str`): the name of the set, can be either `train` or `test`
        """
        if name == 'train':
            split_idx = -7
        elif name == 'test':
            split_idx = -3
        X_PIE = X[:split_idx,:]
        y_PIE = y[:split_idx]
        X_MY = X[split_idx:,:]
        y_MY = y[split_idx:]
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
    `dims` (`list[int]`): list of PCA dimensions to be tested
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
        proj_train_list.append(FaceDataset.split(proj_train_X, train.y, name='train'))
        proj_test_list.append(FaceDataset.split(proj_test_X, test.y, name='test'))
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

def KNNClassification(train: FaceDataset, test: FaceDataset) -> np.ndarray:
    """
    Apply KNN classification on the training set data and predict on the test set

    Parameters
    ----------
    `train` (`FaceDataset`): the train data X to be used to train the KNN classifier
    `test` (`FaceDataset`): the test data to be tested with the KNN classifier
    
    Returns
    -------
    `np.ndarray`: classification error rates with a shape of `[2, 1]`
    """
    # Apply KNN classification
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean').fit(train.X, train.y)
    test_y_PIE_pred = knn.predict(test.X_PIE)
    test_y_MY_pred = knn.predict(test.X_MY)
    # Print classification results
    # print('KNN Classification results on PIE test set:')
    # print(classification_report(test.y_PIE, test_y_PIE_pred))
    # print('KNN Classification results on MY test set:')
    # print(classification_report(test.y_MY, test_y_MY_pred))

    # Collect results
    error_rate = np.zeros((2,1))
    error_rate[0, 0] = (test_y_PIE_pred != test.y_PIE).sum() / test_y_PIE_pred.shape[0]
    error_rate[1, 0] = (test_y_MY_pred != test.y_MY).sum() / test_y_MY_pred.shape[0]

    return error_rate

def KNNClassifications(train_list: list, test_list: list) -> np.ndarray:
    """
    Apply KNN Classifications on a list of train and test sets.

    Parameters
    ----------
    `train_list` (`list[FaceDataset]`): the list of train datasets to be used to train the KNN classifier
    `test_list` (`list[FaceDataset]`): the list of test datasets to be tested with the KNN classifier
    
    Returns
    -------
    `np.ndarray`: classification error rates
    """
    error_rates = np.empty(shape=(2,0), dtype=float)
    # For each dimension to be tested
    for i in range(len(train_list)):
        # Apply KNN classifications and store results
        error_rates = np.append(error_rates, KNNClassification(train_list[i], test_list[i]), axis=1)

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

def applyLDAs(dims: list, train: FaceDataset, test: FaceDataset) -> tuple:
    """
    Apply the train data to fit a series of LDAs with different dimensions and show the reconstructed images

    Parameters
    ----------
    `dims` (`list[int]`): list of LDA dimensions to be tested
    `train` (`FaceDataset`): the train data to be used to fit the LDA algorithm
    `test` (`FaceDataset`): the test data to be transformed by the LDA algorithm
    
    Returns
    -------
    `list[FaceDataset]`: list of train set images after LDA dimensionality reduction
    `list[FaceDataset]`: list of test set images after LDA dimensionality reduction
    """
    # Apply LDA on a list of dimensions
    lda_list = []
    proj_train_list = []
    proj_test_list = []

    for i in range(len(dims)):
        lda_list.append(LinearDiscriminantAnalysis(n_components=dims[i]))
        # Fit LDA on the images
        lda_list[i].fit(train.X, train.y)
        proj_train_X = lda_list[i].fit_transform(train.X, train.y)
        proj_test_X = lda_list[i].transform(test.X)
        proj_train_list.append(FaceDataset.split(proj_train_X, train.y, name='train'))
        proj_test_list.append(FaceDataset.split(proj_test_X, test.y, name='test'))

    return proj_train_list, proj_test_list

def GMMClusterings(train_list: list, train: FaceDataset, n_comps: int=3, show_samples: int=10) -> None:
    """
    Apply GMM Clustering on a list of (preprocessed) training set images

    Parameters
    ----------
    `train_list` (`list[FaceDataset]`): the list of train datasets to be used to train the KNN classifier
    `train` (`FaceDataset`): the original image set to be used to visualize the clusters
    `n_comps` (`int`): the list of test datasets to be tested with the KNN classifier
    `show_samples` (`int`) the number of example results to display after done, `0` for no output, default as `10`
    
    Returns
    -------
    `None`: 
    """
    for i in range(len(train_list)):
        # Fit the train data on a GMM and predict
        gmm = GaussianMixture(n_components=n_comps).fit(train_list[i].X)
        cls_pred = gmm.predict(train_list[i].X)
        print(cls_pred)
        # Randomly pick some images from each clusters to display
        cls_idxs_list = []
        for i in range(n_comps):
            cls_idxs = [j for j in range(cls_pred.shape[0]) if cls_pred[j]==i]
            cls_idxs_list.append(random.sample(cls_idxs, show_samples))

        # Plot some example faces in each cluster
        if show_samples > 0:
            n_rows = n_comps
            n_cols = show_samples
            img_shape = np.array([np.sqrt(train.X.shape[1]), np.sqrt(train.X.shape[1])], dtype=int)
            fig = plt.figure(figsize=(16, 6))
            for i in range(n_rows):
                for j in range(n_cols):
                    ax = fig.add_subplot(n_rows, n_cols, i*n_cols+j+1, xticks=[], yticks=[])
                    ax.imshow(train.X[cls_idxs_list[i][j]].reshape(img_shape), cmap='gray')
                    ax.set_xlabel('%d' % train.y[cls_idxs_list[i][j]])
                    if j == 0:
                        ax.set_ylabel('Cluster %d' % i)

            plt.show()

    return

def SVMClassifications(train_list: list, test_list: list, C_list: list) -> np.ndarray:
    """
    Apply a series of SVM Classifications with different C params on a list of train and test sets.

    Parameters
    ----------
    `train_list` (`list[FaceDataset]`): the list of train datasets to be used to train the KNN classifier
    `test_list` (`list[FaceDataset]`): the list of test datasets to be tested with the KNN classifier
    `C_list` (`list[float]`): the list of C values to be used in SVM Classifiers
    
    Returns
    -------
    `np.ndarray`: classification error rates
    """
    results = []
    for i in range(len(C_list)):
        error_rates = np.empty(shape=(2,0), dtype=float)
        for j in range(len(train_list)) :
            svm = SVC(C=C_list[i], class_weight="balanced").fit(train_list[j].X, train_list[j].y)
            test_y_PIE_pred = svm.predict(test_list[j].X_PIE)
            test_y_MY_pred = svm.predict(test_list[j].X_MY)

            # Print classification results
            # print('SVM Classification results on PIE test set:')
            # print(classification_report(test_list[j].y_PIE, test_y_PIE_pred))
            # print('SVM Classification results on MY test set:')
            # print(classification_report(test_list[j].y_MY, test_y_MY_pred))

            # Collect results
            error_rate = np.zeros((2,1))
            error_rate[0, 0] = (test_y_PIE_pred != test_list[j].y_PIE).sum() / test_y_PIE_pred.shape[0]
            error_rate[1, 0] = (test_y_MY_pred != test_list[j].y_MY).sum() / test_y_MY_pred.shape[0]
            error_rates = np.append(error_rates, error_rate, axis=1)
        
        results.append(error_rates)

    return np.vstack(results)

def main():
    # Display Settings
    show_error_rates = False    # If we want to plot the error rates for all algos
    show_pca_result = False      # If we want to plot the PCA results
    show_pca_samples = 0        # Number of example results to display for PCA, `0` for no output
    show_lda_result = False      # If we want to plot the PCA results
    show_gmm_samples = 0        # Number of example results to display for GMM, `0` for no output

    # Set destination paths
    repo_path = pathlib.Path(__file__).parent.parent.resolve()
    data_path = os.path.join(repo_path, 'data')
    print(data_path)

    # Read 500 images from the train set and all from the test set
    train_set = readImageData(data_path, set='train', num_PIE_imgs=500)
    test_set = readImageData(data_path, set='test')

    
    print('Started Task 1: PCA')
    # Apply PCA on 3D (which also included 2D) and visualize results
    plotPCA3DResults(train_set, show_plot=show_pca_result)

    # Apply PCA on 40, 80, 200 dimensions and show the reconstructed images
    pca_dims = [40, 80, 200]
    proj_train_list, proj_test_list = applyPCAs(pca_dims, train_set, test_set, show_samples=show_pca_samples)
    
    # Apply KNN Classifications on the PCA preprocessed image lists
    pca_error_rates = KNNClassifications(proj_train_list, proj_test_list)

    # Apply KNN Classification on the original images (as a baseline for comparison later)
    pca_dims.append(train_set.X.shape[1])
    baseline_error_rate = KNNClassification(train_set, test_set)
    pca_error_rates = np.append(pca_error_rates, baseline_error_rate, axis=1)
    
    # Visualize KNN classification error rates
    if show_error_rates:
        showErrorRates(pca_dims, pca_error_rates)
    print('Finished Task 1: PCA')

    
    print('Started Task 2: LDA')
    # Apply LDA to reduce the dimensionalities of the original images
    lda_dims = [2, 3, 9]
    proj_train_list, proj_test_list = applyLDAs(lda_dims, train_set, test_set)
    # Visualize results
    if show_lda_result:
        # Plot the projected data onto 2D and 3D scatter plots
        plotProjectedData(proj_train_list[1].X_PIE, proj_train_list[1].X_MY)

    lda_error_rates = KNNClassifications(proj_train_list, proj_test_list)
    if show_error_rates:
        showErrorRates(lda_dims, lda_error_rates)
    print('Finished Task 2: LDA')
    

    # Read the whole train and test set (unlike the first time which we only sampled 500)
    train_set = readImageData(data_path, set='train')
    test_set = readImageData(data_path, set='test')
    # Apply PCA preprocessing on all the images
    pca_dims = [80, 200]
    pca_train_list, pca_test_list = applyPCAs(pca_dims, train_set, test_set, show_samples=0)
    print('PCA Pre-processed %d training images and %d test images' % (pca_train_list[0].X.shape[0], pca_test_list[0].X.shape[0]))
    

    print('Started Task 3: GMM')
    GMMClusterings(pca_train_list, train_set, n_comps=3, show_samples=show_gmm_samples)
    print('Finished Task 3: GMM')


    print('Started Task 4: SVM')
    C_list = [0.01, 0.1, 1]
    svm_error_rates = SVMClassifications(pca_train_list, pca_test_list, C_list=C_list)
    print(svm_error_rates)
    print('Finished Task 4: SVM')
    

    print('Done')
    return

if __name__ == "__main__":
    main()