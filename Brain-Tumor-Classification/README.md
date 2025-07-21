![UTA-DataScience-Logo](https://github.com/dareli/DATA3402.Spring.2024/assets/123596270/0cb941d4-8a3b-4382-9dd0-22c28edbb8a5)

# **DATA 4380 - Computer Vision Project: Brain Tumor Classification from MRI:** 
This repository contains experiments using transfer learning (ResNet50, DenseNet121, MobileNetV2, VGG19, EfficientNetB0) to classify brain tumors using MRI scans from the Kaggle Brain Tumor Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## **Overview** 
- The goal was to classify brain MRI images into four categories: glioma, meningioma, pituitary tumor, or no tumor. I used the grayscale MRI images and turned them into a multiclass image classification task using transfer learning and data augmentation to boost performance. Thebest model (DenseNet121 with augmentation) achieved a micro-averaged ROC AUC score of 0.99, with other top models (MobileNetV2 and ResNet50-Augmented) achieving 0.98 and 0.97.

## **Summary of Work Done**
- Data
  - Type: Medical brain MRI images
  - Input: Grayscale JPG images, 224x224, labeled with 4 classes (glioma, meningioma, pituitary tumor, or no tumor)
  - Size: 5712 (around 1428 in each class) training images, 1311 (around 327 per class) testing images
  - Split: 80% train / 20% test
 
## **Preprocessing & Clean up**
 - Cropped out margins
 - Normalized pixel values to [0, 1]
 - Used ImageDataGenerator for train/val splits and augmentations
 -  Small Augmentations: rotation, zoom, flip, etc.
 
## **Data Visualization**
![Tumor Classes](pngs/tumorimagesindices.png)
These are the cropped and endoded MRIs (Class indices: {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3})


## **Problem Formulation**
- Input: 224x224 MRI brain image (grayscale -> RGB)
- Output:
  - One-hot encoded vector representing one of the four classes: glioma, meningioma, pituitary tumor, no tumor
- Models:
  - DenseNet121 (with augmentation)
  - MobileNetV2 (with augmentation)
  - ResNet50 (with and without augmentation)
  - VGG19 (baseline only)
  - EfficientNetB0 (with augmentation)
- Hyperparameters
  - Epochs = 10
  - Batches = 32
 
## **Training & Challenges**
Model training was done using a mix of a 2021 MacBook Pro (14-inch, base model) and my Windows PC with an AMD Ryzen 7 5700G, 32GB RAM, and a GTX 760 Ti GPU. I an everything in Jupyter Notebooks using TensorFlow/Keras, with the ImageDataGenerator pipeline to handle image loading and augmentation for the grayscale MRI scans.

I started off with a 2 base models (ResNet50 and VGG19) then and also ran experiments with augmented versions of each architecture to evaluate the impact of data augmentation. Augmentations included horizontal/vertical flips, zoom, rotation, and brightness/contrast changes.

Due to limited time and hardware, so I didn’t do full hyperparameter tuning (I had to switch to my PC after my laptop crashed running the VGG19 with augmentation). Training times varied across models and machines, so they're not listed exactly here, but heavier models like VGG19 and ResNet50 generally took longer, while MobileNetV2 trained more quickly. Then all models were evaluated on the same validation/test split using ROC-AUC (micro), and training curves were saved to visualize performance.

## **Performance Comparison** 
### Training Curves
**Baseline VGG19 without augmentation**
![Base VGG19](basevgg19.png)

**Baseline ResNet50 without augmentation**
![Base ResNet50](pngs/baserestnet50.png)

**DenseNet121**  
![DenseNet121 Training](pngs/densenet121_learning_curves.png)

**MobileNetV2**  
![MobileNetV2 Training](pngs/MobileNetV2_learning_curves.png)

**EfficientNetB0**  
![EfficientNetB0 Training](pngs/efficientnetb0_learning_curves.png)

### ROC Curves  
**ResNet50 vs VGG19**  
![ROC ResNet vs VGG](pngs/roc_resnet_vgg_models.png)

**Top 3 Models (DenseNet121, MobileNetV2, ResNet50)**  
![Top 3 ROC](pngs/compare3_roc_curve.png)

**ResNet50 Only**  
![ResNet50 ROC](pngs/resnet50_roc_curve.png)


## **Conclusions**
### AUC Comparison  
![Model AUC Comparison](pngs/model_auc_comparison.png)
Out of all the models I tried, DenseNet121 with augmentation came out on top with an impressive ROC AUC of 0.99. Right behind it were MobileNetV2 (0.98) and ResNet50 with augmentation (0.97), all three showed strong performance and solid learning curves on the validation set.

The baseline models did okay too, VGG19 hit 0.92 (with augmentation, it performed very poorly), while ResNet50 without augmentation lagged a bit at 0.82. The only other underperformer was EfficientNetB0 with augmentation, which surprisingly scored just 0.52, possibly because it didn’t play well with grayscale images or needed more tuning.

Overall, DenseNet121 stood out as the most reliable model, especially with augmentation. Augmentation clearly helped most models (especially ResNet and DenseNet), but it wasn’t a guaranteed boost across the board, some architectures just didn’t respond as well.

## **Future Work**
- Hyperparameter tuning, like adjusting learning rates, batch sizes, or trying regularization methods like dropout or weight decay.
- Improve augmentation by testing out more advanced techniques, like elastic distortions or contrast tweaks
- Balance the classes better, especially if some tumor types are underrepresented in the dataset.
- Try grayscale-optimized layers or architectures that might work better with medical image data out of the box.

## **How to Reproduce Results**
1. Download or clone this repository
2. Download the pre-cleaned dataset directly from this repository (tumor_data_cleaned)
3. Run the notebooks in order:
   - BrainTumorClass_DataLoader.ipynb
   - Brain_Tumor_Class_TrainBaseModel.ipynb
   - BrainTumorClass_TrainBaseModelAugmentation.ipynb
   - Train-EfficientNetB0.ipynb, Train-DenseNet121.ipynb, TrainMobileNetV2.ipynb
   - CompareModels.ipynb
   
### Overview of Files in Repository
- BrainTumorClass_DataLoader.ipynb: Data preprocessing and cleanup
- Brain_Tumor_Class_TrainBaseModel.ipynb: Baseline models training
- BrainTumorClass_TrainBaseModelAugmentation.ipynb: Data augmentation and training
- Train-DenseNet121.ipynb: DenseNet121 model training
- TrainMobileNetV2.ipynb: MobileNetV2 training
- MobileNetV2 training: EfficientNetB0 training
- Train-EfficientNetB0.ipynb: EfficientNetB0 training
- ResNet50_CompareAugmentation.ipynb: Compare ResNet50 with/without augmentation
- CompareModels.ipynb: AUC comparison and ROC plots
- class_indices.json: Maps class names to numeric labels
- pngs (folder): ROC curves and training visualizations

### Data
- The dataset used in this project is the Brain Tumor Classification (MRI) Dataset available on Kaggle.(https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)
  - Cropped and encoded data is available in the repository under tumor_data_cleaned

### Preprocessing and Organization
- Images were resized to (224, 224) to match model input size requirements
- Data augmentation was applied using TensorFlow’s tf.keras.Sequential API (For reference, see BrainTumorClass_DataLoader.ipynb)

### Training
#### Environment:
- Majority of training was performed using Jupyter Notebook on a Windows machine with the following specs: AMD Ryzen 7 5700G, 32GB RAM, and a GTX 760 Ti GPU
#### Training Frameworks:
- Models were implemented using TensorFlow/Keras, with common tools such as NumPy, Pandas, and Matplotlib
- All models were trained using: categorical crossentropy loss, adam optimizer, earlystopping with validation monitoring, Batch size of 32, epochs of 10

### Performance Evaluation
- Metrics Used: Validation Accuracy, ROC AUC Score, Training/Validation Curves
- Final Model Comparison: After all models were trained, performance was evaluated on the validation set and summarized in a horizontal bar chart comparing ROC AUC scores. See CompareModels.ipynb for visualizations and metrics.

## **Citations**
[1] Nickparvar, Masoud. “Brain Tumor MRI Dataset.” Kaggle, 2023.
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

[2] Chollet, François. “Image Classification from Scratch.” Keras Documentation.
https://keras.io/examples/vision/image_classification_from_scratch/

[3] TensorFlow. “Load images from directory.” TensorFlow Guide.
https://www.tensorflow.org/tutorials/load_data/images

[4] Nickparvar, Masoud. “Preprocessing Script for Brain Tumor Classification.” GitHub, 2023.
https://github.com/masoudnick/Brain-Tumor-MRI-Classification/blob/main/Preprocessing.py

[5] Shah, Rajan, et al. “Brain Tumor Classification Using Deep Learning and Transfer Learning.” Journal of Healthcare Engineering, 2023.
https://pmc.ncbi.nlm.nih.gov/articles/PMC10157370/#s3











