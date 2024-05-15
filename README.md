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
After unzip of the downloaded dataset to a folder /RAW_DATA, run the following codes:
```commandline
python ./data_prop/data_preprocessing.py --include_seg --rooddir /RAW_DATA --outdir /TRAIN_DATA_WITH_INTERPOLATION
```
and 
```commandline
python ./data_prop/data_preprocessing.py --include_seg --no_depth_interpolation --rooddir /RAW_DATA --outdir /TEST_DATA_WITHOUT_INTERPOLATION
```
### Step 3. Source Model Training
To train the segmentation U-Net model run the following script with params:
```commandline
python ./source_model_training/main_train_source_model.py --dataroot /TRAIN_DATA_WITH_INTERPOLATION --checkpoints_dir /MODEL_SAVE_FOLDER
```


To perform SaGTTA, run script in sagtta

To perform OptTTA, run script in opttta

To evaluated TTA result, run script test_result.py

If you find the code is useful please cite the paper. Thanks!

[^1]: Some codes are from the OptTTA https://github.com/devavratTomar/OptTTA