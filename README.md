# SaGTTA
### Source code of  "SaGTTA: saliency guided test time augmentation for medical image segmentation across vendor domain shift".[^1]



### Step 1. Configure Conda Environment with the YAML file:

```commandline
conda env create -f saggta.yml
```
and activate the virtual environment
```commandline
conda activate saggta
```
### Step 2. Data Preprocessing
#### The dataset is available at http://niftyweb.cs.ucl.ac.uk/challenge/index.php
#### 2a. Cropping, Normalizing and resampling.
After unzip of the downloaded dataset to a folder /RAW_DATA, run the following codes:

To create interpolated training data set (and validation set):
```commandline
python ./data_prep/data_preprocessing.py --include_seg --rooddir /RAW_DATA --outdir /TRAIN_DATA_WITH_INTERPOLATION
```
To create test data set without interpolation in z axis:
```commandline
python ./data_prep/data_preprocessing.py --include_seg --no_depth_interpolation --rooddir /RAW_DATA --outdir /TEST_DATA_WITHOUT_INTERPOLATION

```
#### 2b. Slice volumne nii.gz files to .npy files:
run the following codes:
```commandline
python ./data_prep/volume_to_numpy_slides.py --rootdir /TRAIN_DATA_WITH_INTERPOLATION --outdir /TRAIN_DATA_WITH_INTERPOLATION_NUMPY_SLICE
```
```commandline
python ./data_prep/volume_to_numpy_slides.py --rootdir /TEST_DATA_WITH_INTERPOLATION --outdir /TESR_DATA_WITH_INTERPOLATION_NUMPY_SLICE

```
### Step 3. Source Model Training
To train the segmentation U-Net model run the following script with params:
```commandline
python ./source_model_training/main_train_source_model.py --dataroot /TESR_DATA_WITH_INTERPOLATION_NUMPY_SLICE --checkpoints_dir /SOURCE_MODEL_SAVE_FOLDER 
```
### Step 4. Perform SaGTTA
```commandline
python ./sagtta/main_sagtta.py --n_augs THE_NUMBER_OF_AUGMENTATIONS --checkpoints_source_free_da /SAGTTA_RESULT_SAVE_FOLDER --checkpoints_source_segmentor /SOURCE_MODEL_SAVE_FOLDER --dataroot /TESR_DATA_WITH_INTERPOLATION_NUMPY_SLICE
```
#### THE_NUMBER_OF_AUGMENTATIONS should be set from 1 to 7. In each setting, the top-3 subpolicies will be selected and saved in the file "OptimalSubpolicy.txt". To ensembling all the result from 1 to 7, please create "CandidateSubpolicy.txt" that contain all top subpolices saved in all "OptimalSubpolicy.txt" files in a new /SAGTTA_RESULT_SAVE_FOLDER. And run main_sagtta.py 

### Step 5. Test SaGTTA Result
```commandline
python ./test_result.py --prediction_path /SAGTTA_RESULT_SAVE_FOLDER --dataroot /TESR_DATA_WITH_INTERPOLATION_NUMPY_SLICE
```
#### Note that /SAGTTA_RESULT_SAVE_FOLDER can be SaGTTA result performed on each settings of THE_NUMBER_OF_AUGMENTATIONS and ensembled predictions.

### Step 6. Perform OptTTA
```commandline
python ./opttta/main_opttta.py --n_augs THE_NUMBER_OF_AUGMENTATIONS --checkpoints_source_free_da /SAGTTA_RESULT_SAVE_FOLDER --checkpoints_source_segmentor /SOURCE_MODEL_SAVE_FOLDER --dataroot /TESR_DATA_WITH_INTERPOLATION_NUMPY_SLICE
```
#### Same Running settings with SaGTTA


If you find the code is useful please cite the paper. Thanks!

[^1]: Some codes are from the OptTTA https://github.com/devavratTomar/OptTTA