import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.spatial import distance
import os
from scipy.stats import ks_2samp, linregress
import scipy.linalg
import scipy
import json
import random
import matplotlib.pyplot as plt





def df_to_array(df):
    '''
    Converts Dataframe to numpy array, extracting the feature names, classes and IDs
    
    :param df: Panda dataframe of features
    :type df: :class:`DataFrame`


    :returns
    :class:`numpy.ndarray` --  (num_features x num_samples)
    :class:`list` --  feature names
    :class:`list` --  classes of images
    :class:`list` --  IDs of images
    '''
    IDs = list(df['ID'])
    classes = list(df['label'])
    df = df.drop(columns=['ID'])
    df = df.drop(columns=['label'])
    return df.to_numpy(), list(df.columns),classes,IDs


def array_to_df(radiomicFeatures3D, feature_names, classes,IDs): #r = radiomics_array object
    '''
    Inverse of df_to_array. Converts array back into a dataframe.
    
    :param radiomicFeatures3D: Array containing features, num_features x num_samples
    :type radiomicFeatures3D: :class:`numpy.ndarray`

    :param feature_names: list containing names of features
    :type feature_names: :class:`list`

    :param classes: list of classes (same order as IDs)
    :type classes: :class:`list`
    
    :param IDs: list of IDs
    :type IDs: :class:`list`


    :returns
    :class:`DataFrame` -- Panda dataframe of features

    '''
    df = pd.DataFrame(radiomicFeatures3D, columns=feature_names)
    df.insert(0, 'label', classes)
    df.insert(0, 'ID', IDs)
    return df

def standardize_features(df):
    '''
    Normalize features of dataframe
        
    :param df: Panda dataframe of features
    :type df: :class:`DataFrame`

    :returns
    :class:`DataFrame` -- Standardized Panda dataframe of features

    '''
    radiomicFeatures3D, feature_names, classes, IDs = df_to_array(df)
    for i in range(np.shape(radiomicFeatures3D)[-1]):
        x = radiomicFeatures3D[:, i]
        #     radiomicFeatures3D[:,i] =  (x-np.min(x))/(np.max(x)-np.min(x))
        radiomicFeatures3D[:, i] = (x - np.mean(x)) / np.std(x)  # Standardize

    df = array_to_df(radiomicFeatures3D, feature_names, classes, IDs)
    return df



def distribution_similarity_KS(dist1, dist2):
    '''
    Calculate the Kolmogorov–Smirnov similarity between 2 distributions

    :param dist1: numpy array with values that make up distribution 1
    :type dist1: :class:`numpy.ndarray`

    :param dist2: numpy array with values that make up distribution 2
    :type dist2: :class:`numpy.ndarray`

    
    :returns
    :class:`float` -- Kolmogorov–Smirnov similarity
    '''
    ks_statistic, _ = ks_2samp(dist1, dist2)
    similarity = 1 - ks_statistic #Normalize
    return similarity

def separate_test(dataset, size_test):
    '''
    Separate a test set from dataset

    :param dataset: Panda dataframe of features
    :type dataset: :class:`DataFrame`

    :param size_test: Number of cases in test set
    :type size_test: :class:`int`

    
    :returns
    :class:`DataFrame` -- Panda dataframe of features for training set
    :class:`DataFrame` -- Panda dataframe of features for test set
    '''
    test_ids = []
    for i in range(size_test):
        test_i = random.choice(range(len(dataset['ID'])))
        test_ids.append(dataset['ID'][test_i])
    testset = dataset[dataset['ID'].isin(test_ids)]
    trainset = dataset[~dataset['ID'].isin(test_ids)]
    return trainset, testset

def load_features(data_path, size_test): 
    '''
    Load radiomics features from JSON file and separate train and test set. 
    Requires 3DRadiomicFeatures.json file.
   
    :param data_path: Path to folder containing radiomics feature json file
    :type data_path: :class:`str`

    :param size_test: Number of cases in test set
    :type size_test: :class:`int`

    
    :returns
    :class:`DataFrame` -- Panda dataframe of features for training set
    :class:`DataFrame` -- Panda dataframe of features for test set
    '''
    
    #df = pd.read_csv(csv_path, index_col=False)
    file_path = os.path.join(data_path, "3DRadiomicfeatures.json")
    with open(file_path, 'r') as file:
        loaded_data = json.load(file)
    # In the section below, we split of the last 4 entries as by default they encode class and test set characteristics
    radiomicFeatures3D = np.array(loaded_data['arr'])
    feature_names = loaded_data['names'][:-4]
    IDs = loaded_data['ids']
    classes = radiomicFeatures3D[:, -4]

    radiomicFeatures3D = radiomicFeatures3D[:, :-4]
    original_df = array_to_df(radiomicFeatures3D, feature_names, classes, IDs)
    original_df_no_borderlines = original_df[original_df['label'] != 2]  # Remove borderlines
    original_df_no_borderlines_no_nans = original_df_no_borderlines.dropna(axis='columns')

    original_df_no_borderlines_no_nans = standardize_features(original_df_no_borderlines_no_nans)

    trainset, testset = separate_test(original_df_no_borderlines_no_nans, size_test)

    return trainset, testset



def load_features_full_dataset(data_path): 
    '''
    Load radiomics features from JSON file and return full dataset. 
    Requires 3DRadiomicFeatures.json file.

    :param data_path: Path to folder containing radiomics feature json file
    :type data_path: :class:`str`

    
    :returns
    :class:`DataFrame` -- Panda dataframe of features
    :class:`numpy.ndarray` -- Array of features from malignant class
    :class:`numpy.ndarray` -- Array of features from benign class
    '''
    #df = pd.read_csv(csv_path, index_col=False)
    file_path = os.path.join(data_path, "3DRadiomicfeatures.json")
    with open(file_path, 'r') as file:
        loaded_data = json.load(file)
    # In the section below, we split of the last 4 entries as by default they encode class and test set characteristics
    radiomicFeatures3D = np.array(loaded_data['arr'])
    feature_names = loaded_data['names'][:-4]
    IDs = loaded_data['ids']
    classes = radiomicFeatures3D[:, -4]
    radiomicFeatures3D = radiomicFeatures3D[:, :-4]
    original_df = array_to_df(radiomicFeatures3D, feature_names, classes, IDs)
    original_df_no_borderlines = original_df[original_df['label'] != 2]  # Remove borderlines
    original_df_no_borderlines_no_nans = original_df_no_borderlines.dropna(axis='columns')

    full_dataset_feats = standardize_features(original_df_no_borderlines_no_nans)


    radiomicFeatures3D, feature_names, classes, IDs = df_to_array(full_dataset_feats)
    
    mal_feats = []
    ben_feats = []
    for i in range(radiomicFeatures3D.shape[-1]):
        feature = radiomicFeatures3D[:, i]
        mal = feature[np.where(np.array(classes) == 1)]
        ben = feature[np.where(np.array(classes) == 0)]
        mal_feats.append(mal)
        ben_feats.append(ben)
        
    return full_dataset_feats, np.reshape(np.array(mal_feats), (np.array(mal_feats).shape[1], np.array(mal_feats).shape[0])), np.reshape(np.array(ben_feats), (np.array(ben_feats).shape[1], np.array(ben_feats).shape[0]))


def load_features_full_dataset_with_IDs(data_path): # Full module and df will be returned
    '''
    Load radiomics features from JSON file and return full dataset. 
    Requires 3DRadiomicFeatures.json file.

    :param data_path: Path to folder containing radiomics feature json file
    :type data_path: :class:`str`

    
    :returns
    :class:`DataFrame` -- Panda dataframe of features
    :class:`numpy.ndarray` -- Array of features from malignant class (1)
    :class:`numpy.ndarray` -- Malignant case IDs
    :class:`numpy.ndarray` -- Array of features from benign class (0)
    :class:`numpy.ndarray` -- Benign case IDs
    '''
    #df = pd.read_csv(csv_path, index_col=False)
    file_path = os.path.join(data_path, "3DRadiomicfeatures.json")
   
    with open(file_path, 'r') as file:
        loaded_data = json.load(file)
    # In the section below, we split of the last 4 entries as by default they encode class and test set characteristics
    radiomicFeatures3D = np.array(loaded_data['arr'])
    print('rad shpae', radiomicFeatures3D.shape)
    feature_names = loaded_data['names'][:-4]
    IDs = loaded_data['ids']
    print('IDs', len(IDs))
        # print(IDs)
    classes = radiomicFeatures3D[:, -4]
    print('classes', classes.shape)
    # Make sure datasplits are based on patient ID

    # Extract only radiomics features
    radiomicFeatures3D = radiomicFeatures3D[:, :-4]
    original_df = array_to_df(radiomicFeatures3D, feature_names, classes, IDs)
    original_df_no_borderlines = original_df[original_df['label'] != 2]  # Remove borderlines
    original_df_no_borderlines_no_nans = original_df_no_borderlines.dropna(axis='columns')

    original_df_no_borderlines_no_nans = standardize_features(original_df_no_borderlines_no_nans)

    full_dataset_feats = original_df_no_borderlines_no_nans
    # print('features', full_dataset_feats)
    radiomicFeatures3D, feature_names, classes, IDs = df_to_array(full_dataset_feats)
    # print('IDs', IDs)
    # Split per class
    IDs_mal = np.array(IDs)[np.where(np.array(classes) == 1)]
    IDs_ben = np.array(IDs)[np.where(np.array(classes) == 0)]
    
    mal_feats = []
    ben_feats = []
    for i in range(radiomicFeatures3D.shape[-1]):
        feature = radiomicFeatures3D[:, i]
        mal = feature[np.where(np.array(classes) == 1)]
        ben = feature[np.where(np.array(classes) == 0)]
        mal_feats.append(mal)
        ben_feats.append(ben)
    # print('mal', mal_feats)
    return full_dataset_feats, np.reshape(np.array(mal_feats), (np.array(mal_feats).shape[1], np.array(mal_feats).shape[0])), IDs_mal, np.reshape(np.array(ben_feats), (np.array(ben_feats).shape[1], np.array(ben_feats).shape[0])), IDs_ben#, np.array(mal_feats), np.array(ben_feats)



def load_features_full_dataset_patient_wise(data_path): # Full module and df will be returned
    '''
    Load radiomics features from JSON file and return full dataset. Checks for unique patient IDs.
    Requires 3DRadiomicFeatures.json file.

    :param data_path: Path to folder containing radiomics feature json file
    :type data_path: :class:`str`

    
    :returns
    :class:`DataFrame` -- Panda dataframe of features
    :class:`numpy.ndarray` -- Unique patient IDs in class 0
    :class:`numpy.ndarray` -- Unique patient IDs in class 1
    :class:`numpy.ndarray` -- Array of features from malignant class
    :class:`numpy.ndarray` -- Malignant case IDs
    :class:`numpy.ndarray` -- Array of features from benign class
    :class:`numpy.ndarray` -- Benign case IDs
    '''
    #df = pd.read_csv(csv_path, index_col=False)
    file_path = os.path.join(data_path, "3DRadiomicfeatures.json")
    csv_path = os.path.join(data_path, "preprocessed_data.csv")
        
    with open(file_path, 'r') as file:
        loaded_data = json.load(file)
    # In the section below, we split of the last 4 entries as by default they encode class and test set characteristics
    radiomicFeatures3D = np.array(loaded_data['arr'])
    feature_names = loaded_data['names'][:-4]
    IDs = loaded_data['ids']
    
        # print(IDs)
    classes = radiomicFeatures3D[:, -4]

    # Make sure datasplits are based on patient ID
    csv_df = pd.read_csv(csv_path, index_col=False)
    patient_IDs_0 = []
    patient_IDs_1 = []
    for index, row in csv_df.iterrows():
        patient_id = row['patient_id']
        
        ID = [v for v in IDs if str(patient_id) in v]
        patient_indexes = [IDs.index(v) for v in ID]
        class_ID = classes[patient_indexes][0]
        if int(class_ID)==0:
            patient_IDs_0.append(patient_id)
        elif int(class_ID)==1:
            patient_IDs_1.append(patient_id)
        else:
            print(f'Something went wrong! Class val: {class_ID}')
            return
    unique_patient_IDs_0 = list(dict.fromkeys(patient_IDs_0))
    unique_patient_IDs_1 = list(dict.fromkeys(patient_IDs_1))


    # Extract only radiomics features
    radiomicFeatures3D = radiomicFeatures3D[:, :-4]
    original_df = array_to_df(radiomicFeatures3D, feature_names, classes, IDs)
    original_df_no_borderlines = original_df[original_df['label'] != 2]  # Remove borderlines
    original_df_no_borderlines_no_nans = original_df_no_borderlines.dropna(axis='columns')

    original_df_no_borderlines_no_nans = standardize_features(original_df_no_borderlines_no_nans)

    full_dataset_feats = original_df_no_borderlines_no_nans

    radiomicFeatures3D, feature_names, classes, IDs = df_to_array(full_dataset_feats)

    # Split per class
    IDs_mal = np.array(IDs)[np.where(np.array(classes) == 1)]
    IDs_ben = np.array(IDs)[np.where(np.array(classes) == 0)]
    
    mal_feats = []
    ben_feats = []
    for i in range(radiomicFeatures3D.shape[-1]):
        feature = radiomicFeatures3D[:, i]
        mal = feature[np.where(np.array(classes) == 1)]
        ben = feature[np.where(np.array(classes) == 0)]
        mal_feats.append(mal)
        ben_feats.append(ben)
    # print('mal', mal_feats)
    return full_dataset_feats, unique_patient_IDs_0, unique_patient_IDs_1, np.reshape(np.array(mal_feats), (np.array(mal_feats).shape[1], np.array(mal_feats).shape[0])), IDs_mal, np.reshape(np.array(ben_feats), (np.array(ben_feats).shape[1], np.array(ben_feats).shape[0])), IDs_ben


def read_MedicalNet_features(data_path, csv_path_0, csv_path_1):
    '''
    Load radiomics features from JSON file and return full dataset. Checks for unique patient IDs.
    Requires 3DRadiomicFeatures.json file to find case IDs.

    :param data_path: Path to folder containing radiomics feature json file
    :type data_path: :class:`str`

    :param csv_path_0: Full path to csv file containing MedicalNet encoded featuresof benign class.
    :type csv_path_0: :class:`str`

    :param csv_path_1: Full path to csv file containing case names of malignant class.
    :type csv_path_1: :class:`str`


    :returns
    :class:`numpy.ndarray` -- Array of features from benign class (0)
    :class:`numpy.ndarray` -- Array of features from malignant class (1)
    :class:`numpy.ndarray` -- Benign case IDs (0)
    :class:`numpy.ndarray` -- Malignant case IDs (1)
    '''
    class_0_feats = np.genfromtxt(csv_path_0, delimiter=',')
    class_1_feats = np.genfromtxt(csv_path_1, delimiter=',')

    print("Class 0 Feature Shape:", class_0_feats.shape)
    print("Class 1 Feature Shape:", class_1_feats.shape)


    file_path = os.path.join(data_path, "3DRadiomicfeatures.json")
    csv_path = os.path.join(data_path, "preprocessed_data.csv")

    with open(file_path, 'r') as file:
        loaded_data = json.load(file)
    # In the section below, we split of the last 4 entries as by default they encode class and test set characteristics
    radiomicFeatures3D = np.array(loaded_data['arr'])
    feature_names = loaded_data['names'][:-4]
    IDs = loaded_data['ids']

    classes = radiomicFeatures3D[:, -4]

    # Extract only radiomics features
    radiomicFeatures3D = radiomicFeatures3D[:, :-4]
    original_df = array_to_df(radiomicFeatures3D, feature_names, classes, IDs)
    original_df_no_borderlines = original_df[original_df['label'] != 2]  # Remove borderlines
    original_df_no_borderlines_no_nans = original_df_no_borderlines.dropna(axis='columns')

    original_df_no_borderlines_no_nans = standardize_features(original_df_no_borderlines_no_nans)

    full_dataset_feats = original_df_no_borderlines_no_nans

    radiomicFeatures3D, feature_names, classes, IDs = df_to_array(full_dataset_feats)

    # Split per class
    IDs_mal = np.array(IDs)[np.where(np.array(classes) == 1)]
    IDs_ben = np.array(IDs)[np.where(np.array(classes) == 0)]
    
    return class_0_feats, class_1_feats, IDs_ben, IDs_mal


def read_MedicalNet_features_FID(csv_path_0, csv_path_1):
    '''
    Load radiomics features from JSON file and return full dataset. Checks for unique patient IDs.
    Does not require 3DRadiomicFeatures.json file.

    :param csv_path_0: Full path to csv file containing case names of benign class.
    :type csv_path_0: :class:`str`

    :param csv_path_1: Full path to csv file containing case names of malignant class.
    :type csv_path_1: :class:`str`


    :returns
    :class:`numpy.ndarray` -- Array of features from benign class (0)
    :class:`numpy.ndarray` -- Array of features from malignant class (1)

    '''
    class_0_feats = np.genfromtxt(csv_path_0, delimiter=',')
    class_1_feats = np.genfromtxt(csv_path_1, delimiter=',')

    
    print("Class 0 Feature Shape:", class_0_feats.shape)
    print("Class 1 Feature Shape:", class_1_feats.shape)
    return class_0_feats, class_1_feats

def FID_metric_calc(class_0_feats, class_1_feats, axis=0):
    '''
    Calculates FID based score of dataset, using the method described in 
    this paper https://link.springer.com/article/10.1007/s00371-020-01922-5

    :param class_0_feats: Array of size [N, X] of encoded image features of class 0. 
                    where N is number of samples and X is number of features per sample
    :type class_0_feats: :class:`numpy.ndarray`

    :param class_1_feats: Array of size [N, X] of encoded image features of class 1
                    where N is number of samples and X is number of features per sample
    :type class_1_feats: :class:`numpy.ndarray`

    :param axis: Specified axis along which to calculate FID score
                    Default: 0
    :type axis: :class:`int`
    
    :returns
    :class:`float` -- FID based score
    '''

    class_0_set_A = class_0_feats[:round(len(class_0_feats)/2)]
    class_0_set_B = class_0_feats[round(len(class_0_feats)/2):]
    class_1_set_A = class_1_feats[:round(len(class_1_feats)/2)]
    class_1_set_B = class_1_feats[round(len(class_1_feats)/2):]


    fid_different = calculate_fid(class_1_feats, class_0_feats, axis)
    fid_class_0 = calculate_fid(class_0_set_A, class_0_set_B, axis)
    fid_class_1 = calculate_fid(class_1_set_A, class_1_set_B, axis)


    if fid_class_0 < fid_class_1:
        FID_ii = fid_class_0
    elif fid_class_0 > fid_class_1:
        FID_ii = fid_class_1
    else:
        FID_ii = fid_class_1
    # print('FID_ii', FID_ii)

    FID_ij = fid_different
    
    ## Calculation of FID based score
    if FID_ii < FID_ij:
        f = 1-FID_ii / FID_ij
    elif FID_ii == FID_ij:
        f = 0
    elif FID_ii > FID_ij:
        f = FID_ij / FID_ii -1

    return f


def calculate_fid(feats1, feats2, axis=0):
    '''
    Calculates FID based score of dataset, using the method described in 
    this paper https://link.springer.com/article/10.1007/s00371-020-01922-5

    :param feats1: Array of size [N, X] of encoded image features of class 0. 
                    where N is number of samples and X is number of features per sample
    :type feats1: :class:`numpy.ndarray`

    :param feats2: Array of size [N, X] of encoded image features of class 1
                    where N is number of samples and X is number of features per sample
    :type feats2: :class:`numpy.ndarray`

    :param axis: Specified axis along which to calculate FID score
                    Default: 0
    :type axis: :class:`int`
    
    :returns
    :class:`float` -- FID based score
    '''
    # calculate mean and covariance statistics
    mu1, sigma1 = feats1.mean(axis=axis), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=axis), np.cov(feats2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)

    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def plot_KS(full_dataset, df,df_test, plot=True):
    '''
    Plots interclass and intraclass KS similarities in bar chart

    :param full_dataset: Panda dataframe of features of full dataset
    :type full_dataset: :class:`DataFrame`

    :param df: Panda dataframe of features of trainset
    :type df: :class:`DataFrame`

    :param df_test: Panda dataframe of features of testset
    :type df_test: :class:`DataFrame`

    :param plot: Boolean to choose wether to make plot or only return values
                    Default: True
    :type plot: :class:`boolean`
    
    :returns
    :class:`numpy.ndarray` -- Malignant feature separation
    :class:`numpy.ndarray` -- Benign feature separation
    
    '''

    train_features = df.drop(columns=['ID', 'label']).to_numpy()
    train_classes = df['label'].to_numpy()

    val_features = df_test.drop(columns=['ID', 'label']).to_numpy()
    val_classes = df_test['label'].to_numpy()
    radiomicFeatures3D, feature_names, classes, IDs = df_to_array(full_dataset)
    def compare_intra_class_vs_inter_class(t_features, t_classes, v_features, v_classes, all_features, _all_classes, func):
        ps = []
        for i in range(t_features.shape[-1]):
            feature = t_features[:, i]
            mal = feature[np.where(t_classes == 1)]
            feature = v_features[:, i]
            mal_v = feature[np.where(v_classes == 1)]
            p = func(mal, mal_v)
            ps.append(p)
        ps2 = []
        for i in range(all_features.shape[-1]):
            feature = all_features[:, i]
            mal = feature[np.where(np.array(_all_classes) == 1)]
            ben = feature[np.where(np.array(_all_classes) == 0)]
            p = func(mal, ben)
            ps2.append(p)
        return ps, ps2
    ps, ps2 = compare_intra_class_vs_inter_class(train_features, train_classes, val_features, val_classes,
                                                 radiomicFeatures3D, classes, distribution_similarity_KS)

    if plot:
        bins = 15
        plt.figure(figsize=(6, 5))

        # Create a subplot with two rows (2x1)

        # Plot the first histogram (ps)
        plt.hist(ps, bins, alpha=0.5, label='Mal vs Mal (train vs. val)')
        plt.hist(ps2, bins, alpha=0.5, label='Ben vs mal (train)')
        plt.legend()
        plt.xlabel("Similarity")
        plt.ylabel("Occurrence")
        plt.grid()
        plt.title("Feature-wise KS-similarity on Ovarian dataset")
        plt.tight_layout()  # Ensure proper spacing between subplots

        plt.show()

        raise ValueError("Plot generated, now exiting...")
    return ps, ps2

def plot_features(df, df_test):
    '''
    Plots interclass and intraclass KS similarities in bar chart

    :param df: Panda dataframe of features of trainset
    :type df: :class:`DataFrame`

    :param df_test: Panda dataframe of features of testset
    :type df_test: :class:`DataFrame`
    
    '''
    full_dataset = pd.concat([df, df_test], ignore_index=True)
    ps_ov,ps2_ov = plot_KS(full_dataset, df, df_test,plot=False)

    bins = 15

    fig, axs = plt.subplots(1, 2, figsize=(10, 2.5))
    ax = axs[0]
    # Create a subplot with two rows (2x1)


    # Plot the first histogram (ps)
    ax.hist(ps_ov, bins, alpha=0.5,  label='Intraclass')
    ax.hist(ps2_ov, bins, alpha=0.5, label='Interclass')
    ax.legend()
    ax.set_xlabel("KS-Similarity")
    ax.set_ylabel("$N$")
    ax.grid()
    ax.text(-0.02,1, chr(65 + 0), transform=ax.transAxes, fontsize=14, weight='bold',
            verticalalignment='bottom', horizontalalignment='right', fontname='Nimbus Roman')
    ax.set_title("Ovarian dataset")

    ###  Temporary sollution to only display ovarian
    ax = axs[1]
    # Create a subplot with two rows (2x1)


    # Plot the first histogram (ps)
    ax.hist(ps_ov, bins, alpha=0.5,  label='Intraclass')
    ax.hist(ps2_ov, bins, alpha=0.5, label='Interclass')
    ax.legend()
    ax.set_xlabel("KS-Similarity")
    ax.set_ylabel("$N$")
    ax.grid()
    ax.text(-0.02,1, chr(65 + 0), transform=ax.transAxes, fontsize=14, weight='bold',
            verticalalignment='bottom', horizontalalignment='right', fontname='Nimbus Roman')
    ax.set_title("Ovarian dataset")

    plt.tight_layout()  # Ensure proper spacing between subplots
    plt.show()


def euc_distance_features(dataset1, dataset2):
    '''
    Calculate Euclidean distance between features of two datasets
        
    :param dataset1: Array of dataset features of dataset 1
    :type dataset1: :class:`numpy.ndarray`

    :param dataset2: Array of dataset features of dataset 2
    :type dataset2: :class:`numpy.ndarray`

    :returns
    :class:`float` -- Euclidean distance between datasets

    '''
    e_dist = distance.euclidean(dataset1,dataset2)
    return e_dist


def SIL_score_calc(class_0_feats, class_1_feats, patient_IDs_0, patient_IDs_1):
    '''
    Calculated Silhouette  score, using the method described in 
    this paper https://link.springer.com/article/10.1007/s00371-020-01922-5

    :param class_0_feats: array of size [N, X] of one class 
                where N is number of samples and X is number of features per sample
    :type class_0_feats: :class:`numpy.ndarray`
    
    :param class_1_feats: array of size [N, X] of second class 
                where N is number of samples and X is number of features per sample
    :type class_1_feats: :class:`numpy.ndarray`

    :param patient_IDs_0: List of patient IDs of class 0
    :type patient_IDs_0: :class:`list`

    :param patient_IDs_1: List of patient IDs of class 1
    :type patient_IDs_1: :class:`list`


    :returns
    :class:`float` -- Silhouette based score
    '''
        
    # Loop over all class 0 samples   
    SIL_scores_0 = []           # 
    for idx_i in range(len(patient_IDs_0)):
        patient_ID_i = patient_IDs_0[idx_i]
        feat_i = class_0_feats[idx_i]

        # Process same class sum
        euc_same = []

        for idx_j in range(len(patient_IDs_0)):
            patient_ID_j = patient_IDs_0[idx_j]
            if patient_ID_j == patient_ID_i:   # If it is the same ID as original then skip
                continue

            feat_j = class_0_feats[idx_j]

            euc_dist_sample = euc_distance_features(feat_i, feat_j)
            
            euc_same.append(euc_dist_sample)

        euc_diff = []

        for idx_j in range(len(patient_IDs_1)):
            patient_ID_j = patient_IDs_1[idx_j]

            if patient_ID_j == patient_ID_i:   # If it is the same ID as original then skip
                print('Something went wrong! The original ID is in the other class list!')
                break

            feat_j = class_1_feats[idx_j]

            euc_dist_sample = euc_distance_features(feat_i, feat_j)
            euc_diff.append(euc_dist_sample)

        a_i = np.average(np.array(euc_same))    # Average distance between sample i and all same-class samples
        b_i = np.average(np.array(euc_diff))    # Average distance between sample i and all different-class samples
        
        # Calculate Silhouette score for one sample
        if a_i < b_i:
            s_i = 1-a_i/b_i
        elif a_i == b_i:
            s_i = 0
        elif a_i > b_i:
            s_i = b_i/a_i -1
        
        
        SIL_scores_0.append(s_i)


    # Loop over all class 1 samples   
    SIL_scores_1 = []           # 
    for idx_i in range(len(patient_IDs_1)):
        patient_ID_i = patient_IDs_1[idx_i]
        feat_i = class_1_feats[idx_i]

        # Process same class sum
        euc_same = []

        for idx_j in range(len(patient_IDs_1)):
            patient_ID_j = patient_IDs_1[idx_j]
            if patient_ID_j == patient_ID_i:   # If it is the same ID as original then skip
                continue

            feat_j = class_1_feats[idx_j]

            euc_dist_sample = euc_distance_features(feat_i, feat_j)
            
            euc_same.append(euc_dist_sample)

        euc_diff = []

        for idx_j in range(len(patient_IDs_0)):
            patient_ID_j = patient_IDs_0[idx_j]

            if patient_ID_j == patient_ID_i:   # If it is the same ID as original then skip
                print('Something went wrong! The original ID is in the other class list!')
                break

            feat_j = class_0_feats[idx_j]

            euc_dist_sample = euc_distance_features(feat_i, feat_j)
            euc_diff.append(euc_dist_sample)

        a_i = np.average(np.array(euc_same))    # Average distance between sample i and all same-class samples
        b_i = np.average(np.array(euc_diff))    # Average distance between sample i and all different-class samples
        
        # Calculate Silhouette score for one sample
        if a_i < b_i:
            s_i = 1-a_i/b_i
        elif a_i == b_i:
            s_i = 0
        elif a_i > b_i:
            s_i = b_i/a_i -1
        else:
            print('Silhuette score calc error!')
        
        
        SIL_scores_1.append(s_i)
    
    
    SIL_final_score = np.average(np.append(np.array(SIL_scores_0), SIL_scores_1))



    return SIL_final_score


def bootstrap_with_blocks(class_0_feats, class_1_feats, patient_IDs_0, patient_IDs_1, function, n_sessions=1000, b_0=0.3, b_1=0.3):
    '''
    Block bootstrapping with function based on samples from 2 classes. 

    :param class_0_feats: array of size [N, X] of one class 
                where N is number of samples and X is number of features per sample
    :type class_0_feats: :class:`numpy.ndarray`
    
    :param class_1_feats: array of size [N, X] of second class 
                where N is number of samples and X is number of features per sample
    :type class_1_feats: :class:`numpy.ndarray`
    
    :param patient_IDs_0: List of patient IDs of class 0
    :type patient_IDs_0: :class:`list`

    :param patient_IDs_1: List of patient IDs of class 1
    :type patient_IDs_1: :class:`list`

    :param function: Function name to use for bootstrapping metrics calculations
    :type function: :class:`function`
    
    :param n_sessions: Number of sessions for bootstrapping
                    Default: 1000
    :type n_sessions: :class:`int`

    :param b_0: Fraction of samples selected for each block of class 0
                    Default: 0.3
    :type b_0: :class:`float`

    :param b_1: fraction of samples selected for each block of class 1
                    Default: 0.3
    :type b_1: :class:`float`
    
    
    :returns
    :class:`numpy.ndarray` -- Array of all bootstrapped scores
    '''
    value_array = np.zeros((n_sessions))
    for i in tqdm(range(n_sessions)):

        class_0_indexes = range(len(class_0_feats))

        idx_class_0 = np.random.choice(class_0_indexes, round(len(class_0_feats)*b_0), replace=False)

        class_0_samples = class_0_feats[idx_class_0-1]
        pat_IDs_0 = patient_IDs_0[idx_class_0-1]

        class_1_indexes = range(len(class_1_feats))

        idx_class_1 = np.random.choice(class_1_indexes, round(len(class_1_feats)*b_1), replace=False)
        class_1_samples = class_1_feats[idx_class_1-1]
        pat_IDs_1 = patient_IDs_1[idx_class_1-1]

        value = function(class_0_samples, class_1_samples, pat_IDs_0, pat_IDs_1)

        value_array[i] = value
    
    return np.array(value_array)



def bootstrap_with_blocks_FID(class_0_feats, class_1_feats, function, n_sessions=1000, b_0=0.3, b_1=0.3):
    '''
    Block bootstrapping with function based on samples from 2 classes. 

    param class_0_feats: array of size [N, X] of one class 
                where N is number of samples and X is number of features per sample
    :type class_0_feats: :class:`numpy.ndarray`
    
    :param class_1_feats: array of size [N, X] of second class 
                where N is number of samples and X is number of features per sample
    :type class_1_feats: :class:`numpy.ndarray`
    
    :param function: Function name to use for bootstrapping metrics calculations
    :type function: :class:`function`
    
    :param n_sessions: Number of sessions for bootstrapping
                    Default: 1000
    :type n_sessions: :class:`int`

    :param b_0: Fraction of samples selected for each block of class 0
                    Default: 0.3
    :type b_0: :class:`float`

    :param b_1: fraction of samples selected for each block of class 1
                    Default: 0.3
    :type b_1: :class:`float`
    
    
    :returns
    :class:`numpy.ndarray` -- Array of all bootstrapped scores
    '''
    value_array = np.zeros((n_sessions))
    for i in tqdm(range(n_sessions)):
        class_0_indexes = range(len(class_0_feats))
        idx_class_0 = np.random.choice(class_0_indexes, round(len(class_0_feats)*b_0), replace=False)
        class_0_samples = class_0_feats[idx_class_0]

        class_1_indexes = range(len(class_1_feats))
        idx_class_1 = np.random.choice(class_1_indexes, round(len(class_1_feats)*b_1), replace=False)
        class_1_samples = class_1_feats[idx_class_1]

        value = function(class_0_samples, class_1_samples)
        value_array[i] = value
    
    return value_array


def bootstrap_with_blocks_patient_wise(class_0_feats, class_1_feats, patient_IDs_0, patient_IDs_1, case_IDs_0, case_IDs_1, function, n_sessions=1000, b_0=0.3, b_1=0.3):
    '''
    Block bootstrapping with function based on samples from 2 classes. 
    This function allows for selecting specific case IDs to include in datasets

    param class_0_feats: array of size [N, X] of one class 
                where N is number of samples and X is number of features per sample
    :type class_0_feats: :class:`numpy.ndarray`
    
    :param class_1_feats: array of size [N, X] of second class 
                where N is number of samples and X is number of features per sample
    :type class_1_feats: :class:`numpy.ndarray`
    
    :param patient_IDs_0: List of patient IDs of class 0
    :type patient_IDs_0: :class:`list`

    :param patient_IDs_1: List of patient IDs of class 1
    :type patient_IDs_1: :class:`list`

    :param case_IDs_0: List of case IDs of class 0
    :type case_IDs_0: :class:`list`

    :param case_IDs_1: List of case IDs of class 1
    :type case_IDs_1: :class:`list`

    :param function: Function name to use for bootstrapping metrics calculations
    :type function: :class:`function`
    
    :param n_sessions: Number of sessions for bootstrapping
                    Default: 1000
    :type n_sessions: :class:`int`

    :param b_0: Fraction of samples selected for each block of class 0
                    Default: 0.3
    :type b_0: :class:`float`

    :param b_1: fraction of samples selected for each block of class 1
                    Default: 0.3
    :type b_1: :class:`float`
    
    
    :returns
    :class:`numpy.ndarray` -- Array of all bootstrapped scores
    '''
    
    value_array = np.zeros((n_sessions))
    for i in range(n_sessions):
        
        rand_pat_IDs_0 = np.random.choice(patient_IDs_0, round(len(patient_IDs_0)*b_0), replace=False)

        class_0_samples = np.empty((0,len(class_0_feats[0])), int)

        for pat_ID in rand_pat_IDs_0:
            ID = [v for v in case_IDs_0 if str(pat_ID) in v]

            patient_indexes = [list(case_IDs_0).index(str(v)) for v in ID]

            for patient_idx in patient_indexes:
                class_0_samp = class_0_feats[patient_idx]
                class_0_samples = np.append(class_0_samples, np.array([class_0_samp]), axis=0)
        
        rand_pat_IDs_1 = np.random.choice(patient_IDs_1, round(len(patient_IDs_1)*b_1), replace=False)
        
        class_1_samples = np.empty((0,len(class_1_feats[0])), int)
        for pat_ID in rand_pat_IDs_1:
            ID = [v for v in case_IDs_1 if str(pat_ID) in v]
            patient_indexes = [list(case_IDs_1).index(str(v)) for v in ID]

            for patient_idx in patient_indexes:
                class_1_samp = class_1_feats[patient_idx]
                class_1_samples = np.append(class_1_samples, [class_1_samp], axis=0)

        value = function(class_0_samples, class_1_samples)
        value_array[i] = value
    return value_array
