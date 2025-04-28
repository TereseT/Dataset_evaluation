from utils import load_features_full_dataset_with_IDs, FID_metric_calc
import time


if __name__ == '__main__':

    # LIDC paths 
    data_p = './NIFTI-LIDC'
    preprocessed_p = './NIFTI-LIDC/NIFTI-Processed_new'
    

    feature_json_path = preprocessed_p + '_FAST'
    full_dataset_features, class_1_features, IDs_class1, class_0_features, IDs_class0 = load_features_full_dataset_with_IDs(feature_json_path)

    start_time = time.time()
    # Process each image in the dataset
    f = FID_metric_calc(class_0_features, class_1_features)
    end_time = time.time()

    elapsed_time = end_time - start_time


    print(f"\nInference Time: {elapsed_time:.4f} seconds")
