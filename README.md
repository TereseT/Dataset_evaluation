# Estimating the difficulty of medical classification tasks using 3D image datasets

This is the code which was used for the dataset evaluation of the conference paper "Estimating the difficulty of medical classification tasks using 3D image datasets", Thornblad et al., 2025 [1]. 

If you find this code useful in your research please cite this paper. 


## Description

Medical deep learning development has seen a rise in recent years. However, for certain applications the model performance remains low compared to other medical applications. To give an early indication of the expected performance for a particular task, developers often start time and resource-consuming benchmark studies, involving training many deep learning models. This work instead aims to create a metric for how challenging a classification task is, based on image features of the dataset. Two dataset difficulty metrics, Silhouette score (SIL) and Fréchet inception distance (FID), are applied to 3D medical image classification datasets, based on radiomic features and deep learning encoded features. These metrics are compared to the performance of two~deep learning models, to estimate each metric's ability to predict dataset difficulty.  This can guide not only the allocation of time and resources on different projects but also further data curation and model development together with clinicians. To promote reproducibility, the code used in this study will be publicly available on GitHub upon acceptance of the paper.


The dataset evaluation methods in this repo are based on the ones presented in the work of Scheidegger et al. [2]

The radiomics feature extraction is based on the OvaCADx_SPIE2025 git repository ( https://github.com/eloyschultz/ovacadx_SPIE2025 ) from the work of Schultz et al. [3]

Deep learning features are extracted using the pretrained MedicalNet models by extracting the features before the segmentation layer and using global average pooling and PCA to reduce the feature dimensionality. 

## Requirements

To use this repository the following packages are required:
Numpy, Scipy, pandas, scikit-learn, SimpleITK

## Use 

The scripts in this repository are for dataset analysis based on extracted image features. To use these scripts encoded image features are required. For this work the script of Eloy [https://github.com/eloyschultz/ovacadx_SPIE2025] was used for radiomic extracted image features and an adaptation of the MedicalNet script [https://github.com/Tencent/MedicalNet] was used for deep learning encoded features. 

To calculate the Fréchet inception distance-based score based on image features, use the jupyter notebook script /Fréchet_inception_distance/FID_calculation.ipynb for calculation with or without block-based bootstrapping. 


To calculate the Silhouette score based on image features, use the jupyter notebook script /Silhouette_score/Silhouette_score_calc_Medical.ipynb for calculation with or without block-based bootstrapping. 

In addition there are scripts for image preprocessing in the main folder. 

[1] "Estimating the difficulty of medical classification tasks using 3D image datasets", Thornblad et al., 2025

[2] Efficient image dataset classification difficulty estimation for predicting deep-learning accuracy, Scheidegger et al., 2020
https://link.springer.com/article/10.1007/s00371-020-01922-5


[3] E. W. R. Schultz, T. A. E. Hellstr¨om, C. H. B. Claessens, A. H. Koch,
J. Nederend, I. Niers-Stobbe, A. Bruining, J. M. J. Piek, P. H. N. d.
With, and F. v. d. Sommen, “Identifying key challenges in ovarian
tumor classification: a comparative study using deep learning and
radiomics,” in Medical Imaging 2025: Computer-Aided Diagnosis,
S. M. Astley and A. Wism¨uller, Eds., vol. 13407. SPIE, 2025, p.
134071W, backup Publisher: International Society for Optics and
Photonics. [Online]. Available: https://doi.org/10.1117/12.3045208
