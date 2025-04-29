from setting import parse_opts 
from datasets.brains18 import BrainS18Dataset
from model import generate_model
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import time
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm





def test(data_loader, model, sets):
    '''
    Run inference on dataset to extract image features from dataset in specified dataset

    :param data_loader: MedicalNet feature maps
    :type data_loader: :class:`torch.utils.data.dataloader.DataLoader`

    :param model: MedicalNet model
    :type model: :class:`torch.nn.parallel.data_parallel.DataParallel`

    :param sets: Parse arguments specifying data paths
    :type sets: :class:`argparse.Namespace`

    
    :returns
    :class: list of tensors -- Extracted image features
    '''
    feats = []
    model.eval() # for testing 
    for batch_id, batch_data in enumerate(data_loader):
        # forward
        volume = batch_data
        if not sets.no_cuda:
            volume = volume.cuda()
        with torch.no_grad():
            probs = model(volume)
            
            
        feats.append(probs)
    
    return feats


def extract_features(feature_maps):
    '''
    Convert MedicalNet feature maps into 2048-dimensional vectors using global average pooling (GAP).

    :param feature_maps: MedicalNet feature maps
    :type feature_maps: :class:`numpy.ndarray`

    :returns
    :class:`numpy.ndarray` -- Reduced features
    '''
    feature_maps = feature_maps.squeeze(1)  # Remove extra dimension -> [N, 2048, 7, 32, 32]
    pooled_features = F.adaptive_avg_pool3d(feature_maps, (1, 1, 1))  # Reduce to [N, 2048, 1, 1, 1]
    return pooled_features.view(feature_maps.shape[0], -1).detach().cpu().numpy()  # [N, 2048]

def pca_feature_reduction(features_0, features_1, n_components=210):
    '''
    Applies PCA to reduce the feature dimensions to `n_components`.

    :param features_0: Feature matrix for class 0 of shape [N_0, 2048]
    :type features_0: :class:`numpy.ndarray`

    :param features_1: Feature matrix for class 1 of shape [N_0, 2048]
    :type features_1: :class:`numpy.ndarray`

    :param n_components: Number of principal components
                        Default 210
    :type n_components: :class:`int`

    :returns
    :class:`numpy.ndarray` -- Reduced features for class 0
    :class:`numpy.ndarray` -- Reduced features for class 1
    :class:`PCA object` -- The trained PCA model
    '''
    all_features = np.vstack((features_0, features_1))  # Combine for PCA fitting
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(all_features)

    # Split back into class-specific arrays
    class_0_feats = reduced_features[:features_0.shape[0], :]
    class_1_feats = reduced_features[features_0.shape[0]:, :]
    
    return class_0_feats, class_1_feats, pca


def get_features(img_list):
    '''
    Extract image features from dataset to extract image features from dataset in specified dataset

    :param img_list: Path to dataset.txt file of data paths
    :type img_list: :class:`str`

    
    :returns
    :class: `numpy.ndarray` -- Extracted image features
    '''
    # data tensor
    testing_data =BrainS18Dataset(sets.data_root, img_list, sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    # testing
    feature_vectors = test(data_loader, net, sets)

    features = np.array(extract_features(feature_vectors[0]))  
    # feature_array = np.array([np.array(feature_vectors[0].cpu())])
    print('shape', features.shape)
    for i in tqdm(range(1, len(feature_vectors))):
        features = np.append(features, extract_features(feature_vectors[i]), axis=0) # Convert to [N_0, 2048]
    print('shape after', features.shape)

    torch.cuda.empty_cache()
    del feature_vectors, testing_data, data_loader

    return features

def dataset_feature_extraction(datasets, large_dataset=False):
    '''
    Extract image features from all datasets to extract image features from dataset in specified dataset

    :param datasets: Parse arguments specifying data paths
    :type datasets: :class:`argparse.Namespace`
    
    :param large_dataset: Boolean option to process dataset in smaller sets
                        Default: False
    :type large_dataset: :class:`bool`

    
    :returns
    :class: `numpy.ndarray` -- Extracted image features of class 0
    :class: `numpy.ndarray` -- Extracted image features of class 1
    '''

    if large_dataset:
        img_list_0_A = os.path.join(datasets.img_list, 'dataset_0_A.txt')
        img_list_0_B = os.path.join(datasets.img_list, 'dataset_0_B.txt')
        img_list_1_A = os.path.join(datasets.img_list, 'dataset_1_A.txt')
        img_list_1_B = os.path.join(datasets.img_list, 'dataset_1_B.txt')

        features_0_A = get_features(img_list_0_A)
        features_0_B = get_features(img_list_0_B)

        features_1_A = get_features(img_list_1_A)
        features_1_B = get_features(img_list_1_B)

        features_0 = np.concatenate((features_0_A, features_0_B), axis=0)
        features_1 = np.concatenate((features_1_A, features_1_B), axis=0)
    else:
        # Primarily use this. If memory overload then use large_dataset=True
        img_list_0 = os.path.join(datasets.img_list, 'dataset_0.txt')
        img_list_1 = os.path.join(datasets.img_list, 'dataset_1.txt')

        features_0 = get_features(img_list_0)
        features_1 = get_features(img_list_1)

    return features_0, features_1




if __name__ == '__main__':
    # settting
    t1 = time.time()
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'

    large_dataset = False       # If memory overload, use large_dataset=True

    # getting model
    checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)
    net.load_state_dict(checkpoint['state_dict'])

    # For large datasets use large_dataset=True to split datasets
    features_0, features_1 = dataset_feature_extraction(sets, large_dataset=large_dataset)

    print('shape after', features_0.shape)
    print('shape after', features_1.shape)

    class_0_feats, class_1_feats, pca_model = pca_feature_reduction(features_0, features_1)
    
    print("Explained variance per component:", pca_model.explained_variance_ratio_)
    print("Cumulative variance:", np.cumsum(pca_model.explained_variance_ratio_))
    # class_0_feats -> [N_0, 1316]
    # class_1_feats -> [N_1, 1316]
    print("Class 0 Feature Shape:", class_0_feats.shape)
    print("Class 1 Feature Shape:", class_1_feats.shape)

    path = './Fréchet_inception_distance/image_features_MedicalNet'
    if not os.path.exists(path):
        os.mkdir(path)
    
    dataset_name = 'liver'
    file_name = f'{dataset_name}_image_features_0.csv'
    save_path_0 = os.path.join(path, file_name)

    np.savetxt(save_path_0, class_0_feats, delimiter=",")
    
    file_name = f'{dataset_name}_image_features_1.csv'
    save_path_1 = os.path.join(path, file_name)

    np.savetxt(save_path_1, class_1_feats, delimiter=",")

    t2 = time.time()

    t_sec = round(t2-t1)
    t_min = int(np.floor(t_sec/60))

    print(f'Total processing time: {t_min} minutes and {t_sec%60} sec')

            