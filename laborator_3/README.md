#  FII ATNN 2025 - Competition 2
Introduction

Welcome to the 2nd ATNN-2025 Competition, a core component of the Advanced Topics in Neural Networks class project. In this challenge, students are tasked with developing machine learning models to achieve and surpass the benchmark scores on the SVHN dataset. 

[Kaggle competition](https://www.kaggle.com/competitions/fii-atnn-2025-competition-2/overview)

---

#  Description

**Objective**

Students are encouraged to explore a variety of techniques, including but not limited to, data preprocessing, data augmentation, test time augmentation (TTA) and hyperparameter finetuning.

- **The model (VGG-13) must NOT be changed.**
- Only 1 notebook run and submitted on Kaggle is considered for the 2nd evaluation criteria.

**Evaluation Criteria**

The homework will be evaluated based on two primary components:

1. **Training experiments (4 points experiments, 4 points report):**
    - Students must experiment with at least 4 different configurations.
    - Only configurations that achieve at least 60% are considered.
    - Students must write a short report in which they summarize the results.
    - The report should contain explanations about the differences between configurations.
    - You must have links to the code used for each configuration (can be the same link).
    - Verbose reports with Information Gain close to 0 will receive 0 points => Keep it concise, relevant.
    - Extend the report only if you have something important to transmit.

2. **Accuracy - only 1 notebook is considered (10 points accuracy):**
    - Achieve at least 65% accuracy (1 point).
    - Achieve at least 68% accuracy (1 point).
    - Achieve at least 70% accuracy (2 points).
    - Achieve at least 75% accuracy (2 points).
    - Achieve at least 77% accuracy (1 point)
    - Achieve at least 78% accuracy (1 point)
    - Achieve at least 79% accuracy (1 point)
    - Achieve at least 80% accuracy (1 point)
    - Achieve at least 81% accuracy (1 bonus point)
    - Achieve at least 82% accuracy (1 bonus point)

3. **Efficiency - only 1 notebook is considered (2 points)**
    - Achieve at least 70% accuracy and have less than 40 minutes of runtime (1 point).
    - Achieve at least 75% accuracy and have less than 40 minutes of runtime (1 point).
    - Achieve at least 80% accuracy and have less than 40 minutes of runtime (1 bonus point).

---

## Project Structure

| File | Description |
|------|--------------|
| `main.ipynb` | The jupiter notebook manager where is all requirements of laborator 3. |
| `test` | The directory where we make mini-tests. |
| `logs` | The directory where we put the logs (best models or history). |

---

## Project steps

In this work, we carried out the following steps: 
- we verified whether the dataset was balanced; 
- designed an image augmentation pipeline; 
- determined the optimal number of worker processes for data loading; 
- implemented validation functions to ensure the correctness of data structure and data types for the VGG13 architecture; 
- defined evaluation metrics to assess model performance; 
- developed a data loader to efficiently transfer training data to VRAM; 
- implemented training callbacks; 
- performed hyperparameter fine-tuning.


## Data aquisition

The data acquisition phase includes both the main dataset used for the target task and auxiliary datasets employed to support regression-based data augmentation. 
The main dataset is '*SVHN*' with 100 classes, while the auxiliary datasets consist of the '*CIFAR-10*' test set and the '*Oxford-IIIT Pet*' test set.

```python
if os.path.exists("/kaggle/input") and os.path.exists("/kaggle/working"):
    print("Running on Kaggle.")
    file_SVHN_test = "/kaggle/input/fii-atnn-2025-competition-2/SVHN_test.pkl"
    file_SVHN_train = "/kaggle/input/fii-atnn-2025-competition-2/SVHN_train.pkl"
else:
    print("Not on Kaggle.")
    file_SVHN_test = "{}/fii-atnn-2025-competition-2/SVHN_test.pkl".format(DS_PATH)
    file_SVHN_train = "{}/fii-atnn-2025-competition-2/SVHN_train.pkl".format(DS_PATH)

with open(file_SVHN_train, "rb") as fd:
    train_in, train_out = list(zip(*pickle.load(fd)))
    train_in, train_out = np.array(train_in, dtype=np.uint8), np.array(train_out, dtype=np.uint16)

with open(file_SVHN_test, "rb") as fd:
    test_in,  test_out  = list(zip(*pickle.load(fd)))
    test_in,  test_out  = np.array(test_in, dtype=np.uint8),  np.array(test_out, dtype=np.uint16)

# data aquisition, main dataset
svhn_train_ds = RowDataset(dict(inputs=train_in, targets=train_out), name="SVHN train")
svhn_test_ds  = RowDataset(dict(inputs=test_in, targets=test_out), name="SVHN test")

# data aquisition help dataset
cifar10_test = datasets.CIFAR10(root=DS_PATH, train=False, transform=None, download=True)
oxfordIIIPet_test = datasets.OxfordIIITPet(root=DS_PATH, split="test", 
                             target_types="category", 
                             transform=None, download=True)

```
Below, we illustrate the data packaging procedure. The '*meta_ds*' object is a dictionary containing the following 'keys': 'data_reader', which specifies the class responsible for loading data from memory and returning an (inputs, targets) tuple; 'size', which indicates the number of training samples provided by the data reader; and 'num_classes', which defines the number of classes represented in the dataset.
```python
# prepare meta data for training
svhn_train_meta_ds = dict(data_reader=svhn_train_ds, size=50000, num_classes=NUM_CLASSES)
svhn_test_meta_ds  = dict(data_reader=svhn_test_ds,  size=10000, num_classes=NUM_CLASSES)
# prepare meta data for help data
meta_cifar10 = dict(data_reader=cifar10_test, size=10000, num_classes=10)
meta_oxfordIIIPet = dict(data_reader=oxfordIIIPet_test, size=3669, num_classes=37)
# pack help datasset
help_metads = dict(cifar10=meta_cifar10, oxfordIIIPet=meta_oxfordIIIPet)
```

## Data augmentation
The '*get_init_transform*' function converts input images from NumPy format (H, W, C) to tensor format (C, H, W) and spatially centers them so that all samples conform to a uniform size of IMAGE_SIZE.
This function is applied to both the training and test datasets. For the validation dataset, Z-score normalization is performed using the dataset’s per-channel mean and standard deviation.
```python
def get_init_transform(train, image_size, mean, std):
    transform = [ # image->tensor->resize->make square-> 
            # if use 'ToImage' tensor should be numpy array!!!
            v2.ToImage(), # data are transorm to torch tensor in Dataset manager, tensor should be numpy array!!!
            v2.Resize(
                size=int(image_size),),
            v2.CenterCrop(image_size),
        ]
    # For train=False, Z-score normalization is performed using the dataset’s per-channel mean and standard deviation.
    # For train=True, do nothing
    if (train == False):
        transform.extend([
                v2.ToDtype(torch.float32, scale=False), # scale True normalized
                v2.Normalize(mean=mean, std=std, inplace=True),
            ])
    # We use the inplace flag because we can safely change the tensors inplace when normalize is used.
    return v2.Compose(transform)
```
The '*get_transforms*' function is applied exclusively to the training data. It performs a set of image augmentation operations, each selected at random with equal probability. These operations include:
- Data diversification through geometric transformations such as rotation, translation, scaling, mirroring, cropping, and shifting. These techniques artificially expand the dataset and improve model generalization by reducing overfitting.
- Color and illumination variations, including adjustments to brightness, contrast, saturation, and hue. This augmentation strategy increases data diversity, enhances feature discriminability, and further mitigates overfitting.
- Noise injection, which improves the model’s robustness to real-world acquisition conditions.
- Noise reduction, using filters such as Gaussian smoothing to remove artifacts that could negatively affect model learning.
- Random erasing, where randomly selected regions of an image are removed to encourage the model to rely on global contextual information rather than local cues.
```python
def get_transforms(image_size: int, mean, std):
    random_choice = v2.RandomChoice([
        v2.RandomPerspective(
                    distortion_scale=0.15, # controls how much each corner can move. 
                    p=1.0),                # probability of applying the effect
        v2.RandomRotation(degrees=30),     # rotates an image with random angle
        v2.RandomAffine(
                    degrees=30,             # rotation ±30
                    translate=(0.15, 0.15), # horizontal/vertical translation as fraction of image
                    scale=(0.75, 1.05),     # scale factor
                    shear=10),              # shear angle ±10°
        v2.RandomCrop(
                    size=image_size,   # height & width of crop
                    padding=4),        # pixels to pad around the image
        v2.RandomResizedCrop(
                    size=image_size,
                    scale=(0.75, 1.),  # range of area proportion to crop from the original image
                    ratio=(0.8,  1.)), # range of aspect ratio (width/height)
        v2.RandomAdjustSharpness(
                    sharpness_factor=1.5, # controls the degree of sharpness; ( >1 sharpened; <1 slightly blurred)
                    p=1.),                      # probability of applying the transform
        v2.RandomAutocontrast(p=1.), # probability of applying the transform
        v2.RandomEqualize( # histogram of pixel values
                    p=1.), # probability of applying the transform
        v2.ColorJitter(  # randomly changes the brightness, contrast, saturation, and hue
                    brightness=0.5, # factor to change brightness
                    contrast=0.3,   # factor to change contrast
                    saturation=0.3, # factor to change saturation
                    hue=0.3,),      # factor to change hue
        v2.GaussianBlur(  # applies a Gaussian blur
                    kernel_size=(7, 7), # size of the Gaussian kernel
                    # standard deviation of the Gaussian kernel; a float or tuple (min, max) for random sampling
                    sigma=(0.1, 5.)),   # how to handle image borders
        v2.RandomErasing(
                    scale=(0.01, 0.15), # range of area ratio to erase (relative to image area)
                    value=10,           # fill value: single number, tuple, or 'random'
                    inplace=False,      # whether to erase in place or return a new image
                    p=1.),              # probability of applying the transform
        v2.Grayscale(num_output_channels=3), # number of channels in output image: 1 or 3
        v2.RandomHorizontalFlip(),
        v2.Identity(),  # returns the input image unchanged
    ])
    transforms = v2.Compose([random_choice,
        v2.ToDtype(torch.float32, scale=False), # converts uint8 [0,255] -> float32 [0,1]
        v2.Normalize(mean=mean, std=std, inplace=True),
    ])
    return transforms
```
The following two functions implement advanced regression strategies: '*get_all_transforms_smoothing*' and '*get_all_transforms_onehot*'. The '*get_all_transforms_smoothing*' function applies the CutMix, MixUp, and Label Smoothing techniques, with each method selected according to a predefined class-wise probability 𝑝. 
The '*get_all_transforms_onehot*' function follows the same structure but replaces Label Smoothing with a One-Hot Target transformation, which operates by applying a one-hot encoding exclusively to the label inputs while leaving the input data unchanged.
```python
def get_all_transforms_smothing(num_classes: int = 10, smooth_size=0.13, p=None):
    transforms = v2.RandomChoice([
                v2.CutMix(num_classes=num_classes),  # See the CutMix paper
                v2.MixUp(num_classes=num_classes),   # See the MixUp paper
                LabelSmoothing(num_classes=num_classes, smooth_size=smooth_size),  # 
            ], p)
    return transforms

def get_all_transforms_onehot(num_classes: int = 10, p=None):
    transforms = v2.RandomChoice([
                v2.CutMix(num_classes=num_classes),  # See the CutMix paper
                v2.MixUp(num_classes=num_classes),   # See the MixUp paper
                OneHotTarget(num_classes=num_classes),  # 
            ], p)
    return transforms
```


## Analysis of the best results

### Analisis SVHN, with next configuration, run time ~40 min
Below is the AppendHelpDatasets pipeline, which incorporates auxiliary datasets into the main dataset; however, in this configuration, only the primary dataset is retained at the output.
```python

train_ds = AppendHelpDatasets(svhn_train_meta_ds, help_metads=None, 
                              transform=get_init_transform(True,  IMAGE_SIZE, mean, std), 
                              help_from=0.1)
test_ds  = AppendHelpDatasets(svhn_test_meta_ds, help_metads=None, 
                              transform=get_init_transform(False, IMAGE_SIZE, mean, std), 
                              help_from=0.1)

# select the best number workers
test_dl  = DataLoader(test_ds , batch_size=BATCH_SIZE, shuffle=False, num_workers=6, drop_last=False)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=19, drop_last=True)
```

```python
EPOCH = 300
BATCH_SIZE = 128

model = VGG13(3,  train_ds.num_classes) # train_ds.num_classes get number of classes
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda")
optimizer    = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)
```
Tab 1) Inference speed test analisis, using SVHN as dataset
| Device |     Dtype      | TTA Type |        Model Type       | Accuracy |      Elapsed       |
| :----: | :------------: | :------: | :---------------------: | :------: | :----------------: |
|  cuda  | torch.bfloat16 |  basic   |           raw           |  0.6868  | 0.5282 |
|  cuda  | torch.bfloat16 |  basic   |         scripted        |  0.6868  | 0.5398 |
|  cuda  | torch.bfloat16 |  basic   |         scripted        |  0.6868  | 0.5063 |
|  cuda  | torch.bfloat16 |  basic   |          traced         |  0.6868  | 0.5340 |
|  cuda  | torch.bfloat16 |  basic   |          traced         |  0.6868  | 0.5263 |
|  cuda  | torch.bfloat16 |  basic   |          frozen         |  0.6872  | 0.5120 |
|  cuda  | torch.bfloat16 |  basic   |          frozen         |  0.6872  | 0.5110 |
|  cuda  | torch.bfloat16 |  basic   | optimized_for_inference |  0.6866  | **0.6414** |
|  cuda  | torch.bfloat16 |  basic   | optimized_for_inference |  0.6866  | **0.6512** |
|  cuda  | torch.bfloat16 |  basic   |         compiled        |  0.6868  | 0.5129 |
|  cuda  | torch.bfloat16 |  basic   |         compiled        |  0.6868  | 0.5701 |
| :----: | :------------: | :------: | :---------------------: | :------: | :----------------: |
|  cuda  | torch.float16  |  basic   |           raw           |  0.6872  | 0.5379 |
|  cuda  | torch.float16  |  basic   |         scripted        |  **0.6881**  | 0.549  |
|  cuda  | torch.float16  |  basic   |         scripted        |  **0.6881**  | 0.5347 |
|  cuda  | torch.float16  |  basic   |          traced         |  0.6872  |  0.5557  |
|  cuda  | torch.float16  |  basic   |          traced         |  0.6872  | 0.5432 |
|  cuda  | torch.float16  |  basic   |          frozen         |  0.6873  |  0.5200  |
|  cuda  | torch.float16  |  basic   |          frozen         |  0.6873  | 0.5222 |
|  cuda  | torch.float16  |  basic   | optimized_for_inference |  0.6873  | **0.6348** |
|  cuda  | torch.float16  |  basic   | optimized_for_inference |  0.6873  |  **0.6289**   |
|  cuda  | torch.float16  |  basic   |         compiled        |  0.6872  | 0.5473 |
|  cuda  | torch.float16  |  basic   |         compiled        |  0.6872  | 0.5455 |
| :----: | :------------: | :------: | :---------------------: | :------: | :----------------: |
|  cuda  | torch.float32  |  basic   |           raw           |  0.6872  | 0.5467 |
|  cuda  | torch.float32  |  basic   |         scripted        |  0.6872  | 0.5931 |
|  cuda  | torch.float32  |  basic   |         scripted        |  0.6872  | 0.5869 |
|  cuda  | torch.float32  |  basic   |          traced         |  0.6872  | 0.5441 |
|  cuda  | torch.float32  |  basic   |          traced         |  0.6872  | 0.5695 |
|  cuda  | torch.float32  |  basic   |          frozen         |  0.6873  | 0.5642 |
|  cuda  | torch.float32  |  basic   |          frozen         |  0.6873  | 0.5518 |
|  cuda  | torch.float32  |  basic   | optimized_for_inference |  0.6873  | 0.5633 |
|  cuda  | torch.float32  |  basic   | optimized_for_inference |  0.6873  | 0.5613 |
|  cuda  | torch.float32  |  basic   |         compiled        |  0.6872  | 0.5639 |
|  cuda  | torch.float32  |  basic   |         compiled        |  0.6872  | 0.5508 |
| :----: | :------------: | :------: | :---------------------: | :------: | :----------------: |
|  cpu   | torch.bfloat16 |  basic   |           raw           |  0.6872  |  5.7370  |
|  cpu   | torch.bfloat16 |  basic   |         scripted        |  0.6872  |  5.6572  |
|  cpu   | torch.bfloat16 |  basic   |         scripted        |  0.6872  | 5.4923 |
|  cpu   | torch.bfloat16 |  basic   |          traced         |  0.6872  |  5.7662  |
|  cpu   | torch.bfloat16 |  basic   |          traced         |  0.6872  |  5.8958  |
|  cpu   | torch.bfloat16 |  basic   |          frozen         |  0.6872  |   5.3145    |
|  cpu   | torch.bfloat16 |  basic   |          frozen         |  0.6872  |    4.7499     |
|  cpu   | torch.bfloat16 |  basic   | optimized_for_inference |  0.6872  | **3.9749**  |
|  cpu   | torch.bfloat16 |  basic   | optimized_for_inference |  0.6872  |   **4.1093**   |
|  cpu   | torch.bfloat16 |  basic   |         compiled        |  0.6872  |   6.2727    |
|  cpu   | torch.bfloat16 |  basic   |         compiled        |  0.6872  |   5.7564    |
| :----: | :------------: | :------: | :---------------------: | :------: | :----------------: |
|  cpu   | torch.float16  |  basic   |           raw           |  0.6872  |   6.4239    |
|  cpu   | torch.float16  |  basic   |         scripted        |  0.6872  |   5.7941   |
|  cpu   | torch.float16  |  basic   |         scripted        |  0.6872  |    6.1803    |
|  cpu   | torch.float16  |  basic   |          traced         |  0.6872  |  5.8396  |
|  cpu   | torch.float16  |  basic   |          traced         |  0.6872  |   6.2583   |
|  cpu   | torch.float16  |  basic   |          frozen         |  0.6872  |  5.2472   |
|  cpu   | torch.float16  |  basic   |          frozen         |  0.6872  |    4.8446    |
|  cpu   | torch.float16  |  basic   | optimized_for_inference |  0.6872  |  **4.3112**   |
|  cpu   | torch.float16  |  basic   | optimized_for_inference |  0.6872  |  **4.3347**  |
|  cpu   | torch.float16  |  basic   |         compiled        |  0.6872  |   5.5943   |
|  cpu   | torch.float16  |  basic   |         compiled        |  0.6872  | 5.9510  |
| :----: | :------------: | :------: | :---------------------: | :------: | :----------------: |
|  cpu   | torch.float32  |  basic   |           raw           |  0.6872  |   6.4493   |
|  cpu   | torch.float32  |  basic   |         scripted        |  0.6872  |  5.8374   |
|  cpu   | torch.float32  |  basic   |         scripted        |  0.6872  |  6.2957   |
|  cpu   | torch.float32  |  basic   |          traced         |  0.6872  |  5.9750  |
|  cpu   | torch.float32  |  basic   |          traced         |  0.6872  | 6.50428 |
|  cpu   | torch.float32  |  basic   |          frozen         |  0.6872  |  4.9339  |
|  cpu   | torch.float32  |  basic   |          frozen         |  0.6872  |   4.9587   |
|  cpu   | torch.float32  |  basic   | optimized_for_inference |  0.6872  |  **4.3657**   |
|  cpu   | torch.float32  |  basic   | optimized_for_inference |  0.6872  |   **4.2749**   |
|  cpu   | torch.float32  |  basic   |         compiled        |  0.6872  |  6.0212  |
|  cpu   | torch.float32  |  basic   |         compiled        |  0.6872  | 5.4899 |

Based on the results reported in Table 1, execution time does not consistently improve under model optimization. 
Specifically, the '*optimized_for_inference*' method on the CUDA device exhibits higher execution time, whereas on the CPU it achieves the best performance. 
From an accuracy perspective, none of the optimization methods impact accuracy on the CPU. 
In contrast, on the CUDA device, the '*scripted*' optimization method yields the highest accuracy when using the `torch.float16` data type.
When using the `torch.float16` and `torch.bfloat16` data types, a slight reduction in processing time can be observed.

Tab 2) Test-Time Augmentation, using SVHN as dataset
| Device |     Dtype      |         TTA Type        | Model Type | Accuracy | Elapsed |
| :----: | :------------: | :---------------------: | :--------: | :------: | :-----: |
|  cuda  | torch.bfloat16 |          basic          |  scripted  |  0.6868  |  0.6201 |
|  cuda  | torch.bfloat16 |          mirror         |  scripted  |  0.6877  |  0.7215 |
|  cuda  | torch.bfloat16 |        translate        |  scripted  |  **0.6939**  |  0.6903 |
|  cuda  | torch.bfloat16 | mirroring_and_translate |  scripted  |  **0.6938**  |  1.2706 |
|  cuda  | torch.bfloat16 |          rotate         |  scripted  |  0.2988  |  0.7198 |
|  cuda  | torch.bfloat16 |           gray          |  scripted  |  0.6521  |  0.6425 |
|  cuda  | torch.bfloat16 |    adjust_brightness    |  scripted  |  0.4611  |  0.6994 |
|  cuda  | torch.bfloat16 |     adjust_contrast     |  scripted  |  0.4688  |  0.6993 |
|  cuda  | torch.bfloat16 |    adjust_saturation    |  scripted  |  0.4598  |  0.7275 |
|  cuda  | torch.bfloat16 |        adjust_hue       |  scripted  |  0.5254  |  0.7482 |
|  cuda  | torch.bfloat16 |           mixt          |  scripted  |  0.5623  |  1.8984 |
| :----: | :------------: | :---------------------: | :--------: | :------: | :-----: |
|  cuda  | torch.float16  |          basic          |  scripted  |  0.6881  |  0.5752 |
|  cuda  | torch.float16  |          mirror         |  scripted  |  0.6874  |  0.7202 |
|  cuda  | torch.float16  |        translate        |  scripted  |  **0.6936**  |  0.7068 |
|  cuda  | torch.float16  | mirroring_and_translate |  scripted  |  **0.6937**  |  1.3069 |
|  cuda  | torch.float16  |          rotate         |  scripted  |  0.2991  |  0.7405 |
|  cuda  | torch.float16  |           gray          |  scripted  |  0.6519  |  0.6395 |
|  cuda  | torch.float16  |    adjust_brightness    |  scripted  |  0.4617  |  0.7187 |
|  cuda  | torch.float16  |     adjust_contrast     |  scripted  |  0.4695  |  0.7317 |
|  cuda  | torch.float16  |    adjust_saturation    |  scripted  |  0.4603  |  0.7293 |
|  cuda  | torch.float16  |        adjust_hue       |  scripted  |  0.5255  |  0.7589 |
|  cuda  | torch.float16  |           mixt          |  scripted  |  0.5623  |  1.961  |
| :----: | :------------: | :---------------------: | :--------: | :------: | :-----: |
|  cuda  | torch.float32  |          basic          |  scripted  |  0.6872  |  0.631  |
|  cuda  | torch.float32  |          mirror         |  scripted  |  0.6868  |  0.8436 |
|  cuda  | torch.float32  |        translate        |  scripted  |  **0.6934**  |  0.727  |
|  cuda  | torch.float32  | mirroring_and_translate |  scripted  |  **0.694**   |  1.6449 |
|  cuda  | torch.float32  |          rotate         |  scripted  |  0.4544  |  0.8507 |
|  cuda  | torch.float32  |           gray          |  scripted  |  0.6517  |  0.7208 |
|  cuda  | torch.float32  |    adjust_brightness    |  scripted  |  0.4615  |  0.8419 |
|  cuda  | torch.float32  |     adjust_contrast     |  scripted  |  0.4698  |  0.8412 |
|  cuda  | torch.float32  |    adjust_saturation    |  scripted  |  0.4603  |  0.8461 |
|  cuda  | torch.float32  |        adjust_hue       |  scripted  |  0.5253  |   0.88  |
|  cuda  | torch.float32  |           mixt          |  scripted  |  0.5616  |  2.354  |

Based on the results obtained under Test-Time Augmentation (TTA) Tab 2, several conclusions can be drawn. The most efficient TTA methods are '*translate*' and '*mirroring_and_translate*'. The '*mixt*' method represents a combination of the following techniques: '*translate*', '*adjust_brightness*', '*adjust_contrast*', '*adjust_saturation*', and '*adjust_hue*'. The results indicate that image rotation and color-range modifications have a negative impact on accuracy. When comparing outcomes after applying TTA, the choice of data type has only a marginal effect on accuracy. Overall, the best trade-off between accuracy and processing time is achieved by using the model with data type `torch.float16` in combination with the '*translate*' TTA method.


### Analisis SVHN + CIFAR10 + OxfordIIITPet, with next configuration, run time ~40 min
As a data augmentation strategy, I extended the existing methods by incorporating samples from the CIFAR-10 and OxfordIIITPet dataset into the training data. 
For labeling, a counter was used such that each time an image was selected from CIFAR-10 or OxfordIIITPet, the corresponding counter value was assigned as its label.
PS: main datasset is SVHN
```python

train_ds = AddVirtualDatasets(svhn_train_meta_ds, help_metads=help_metads, 
                              transform=get_init_transform(True,  IMAGE_SIZE, mean, std), 
                              # percent of number of auxiliari dataset from main dataset; 50000*0.1 = 5000
                              # size of final dataset is 55000 samples
                              help_from=0.1)
test_ds  = AddVirtualDatasets(svhn_test_meta_ds, help_metads=None, 
                              transform=get_init_transform(False, IMAGE_SIZE, mean, std), 
                              help_from=0)

# select the best number workers
test_dl  = DataLoader(test_ds , batch_size=BATCH_SIZE, shuffle=False, num_workers=3, drop_last=False)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=21, drop_last=False)
```

```python
EPOCH = 300
BATCH_SIZE = 128

model = VGG13(3,  train_ds.num_classes) # train_ds.num_classes get number of classes
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda")
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-9)
```
Tab 3) Inference speed test analisis, using SVHN + CIFAR10 + OxfordIIITPet as dataset
| Device |     Dtype      | TTA Type |        Model Type       | Accuracy | Elapsed |
| :----: | :------------: | :------: | :---------------------: | :------: | :-----: |
|  cuda  | torch.bfloat16 |  basic   |           raw           |  0.7092  |  0.5752 |
|  cuda  | torch.bfloat16 |  basic   |         scripted        |  0.7092  |  0.5596 |
|  cuda  | torch.bfloat16 |  basic   |         scripted        |  0.7092  |  0.5579 |
|  cuda  | torch.bfloat16 |  basic   |          traced         |  0.7092  |  0.6119 |
|  cuda  | torch.bfloat16 |  basic   |          traced         |  0.7092  |  0.5952 |
|  cuda  | torch.bfloat16 |  basic   |          frozen         |  0.709   |  0.5604 |
|  cuda  | torch.bfloat16 |  basic   |          frozen         |  0.709   |  0.5636 |
|  cuda  | torch.bfloat16 |  basic   | optimized_for_inference |  0.7093  |  0.6843 |
|  cuda  | torch.bfloat16 |  basic   | optimized_for_inference |  0.7093  |  0.6781 |
|  cuda  | torch.bfloat16 |  basic   |         compiled        |  0.7087  |  2.4234 |
|  cuda  | torch.bfloat16 |  basic   |         compiled        |  0.7091  |  1.4761 |
| :----: | :------------: | :------: | :---------------------: | :------: | :-----: |
|  cuda  | torch.float16  |  basic   |           raw           |  0.7093  |  0.5852 |
|  cuda  | torch.float16  |  basic   |         scripted        |  0.7088  |  0.5793 |
|  cuda  | torch.float16  |  basic   |         scripted        |  0.7088  |  0.5829 |
|  cuda  | torch.float16  |  basic   |          traced         |  0.7093  |  0.6054 |
|  cuda  | torch.float16  |  basic   |          traced         |  0.7093  |  0.5941 |
|  cuda  | torch.float16  |  basic   |          frozen         |  0.7094  |  0.5665 |
|  cuda  | torch.float16  |  basic   |          frozen         |  0.7094  |  0.5862 |
|  cuda  | torch.float16  |  basic   | optimized_for_inference |  0.7092  |  0.6903 |
|  cuda  | torch.float16  |  basic   | optimized_for_inference |  0.7092  |  0.6705 |
|  cuda  | torch.float16  |  basic   |         compiled        |  0.7092  |  1.7658 |
|  cuda  | torch.float16  |  basic   |         compiled        |  0.7092  |  1.5414 |
| :----: | :------------: | :------: | :---------------------: | :------: | :-----: |
|  cuda  | torch.float32  |  basic   |           raw           |  0.7093  |  0.6246 |
|  cuda  | torch.float32  |  basic   |         scripted        |  0.7093  |  0.6184 |
|  cuda  | torch.float32  |  basic   |         scripted        |  0.7093  |  0.6232 |
|  cuda  | torch.float32  |  basic   |          traced         |  0.7093  |  0.5904 |
|  cuda  | torch.float32  |  basic   |          traced         |  0.7093  |  0.6421 |
|  cuda  | torch.float32  |  basic   |          frozen         |  0.7094  |  0.6265 |
|  cuda  | torch.float32  |  basic   |          frozen         |  0.7094  |  0.6018 |
|  cuda  | torch.float32  |  basic   | optimized_for_inference |  0.7094  |  0.6342 |
|  cuda  | torch.float32  |  basic   | optimized_for_inference |  0.7094  |  0.6211 |
|  cuda  | torch.float32  |  basic   |         compiled        |  0.7094  |  1.3408 |
|  cuda  | torch.float32  |  basic   |         compiled        |  0.7094  |  1.3604 |
| :----: | :------------: | :------: | :---------------------: | :------: | :-----: |
|  cpu   | torch.bfloat16 |  basic   |           raw           |  0.7094  |  6.2464 |
|  cpu   | torch.bfloat16 |  basic   |         scripted        |  0.7094  |  5.3362 |
|  cpu   | torch.bfloat16 |  basic   |         scripted        |  0.7094  |  5.6353 |
|  cpu   | torch.bfloat16 |  basic   |          traced         |  0.7094  |  6.161  |
|  cpu   | torch.bfloat16 |  basic   |          traced         |  0.7094  |  6.3695 |
|  cpu   | torch.bfloat16 |  basic   |          frozen         |  0.7094  |  5.1406 |
|  cpu   | torch.bfloat16 |  basic   |          frozen         |  0.7094  |  5.6637 |
|  cpu   | torch.bfloat16 |  basic   | optimized_for_inference |  0.7094  |  4.0182 |
|  cpu   | torch.bfloat16 |  basic   | optimized_for_inference |  0.7094  |  4.0498 |
|  cpu   | torch.bfloat16 |  basic   |         compiled        |  0.7094  |  5.6101 |
|  cpu   | torch.bfloat16 |  basic   |         compiled        |  0.7094  |  5.9544 |
| :----: | :------------: | :------: | :---------------------: | :------: | :-----: |
|  cpu   | torch.float16  |  basic   |           raw           |  0.7094  |  5.9157 |
|  cpu   | torch.float16  |  basic   |         scripted        |  0.7094  |  5.4907 |
|  cpu   | torch.float16  |  basic   |         scripted        |  0.7094  |  5.4827 |
|  cpu   | torch.float16  |  basic   |          traced         |  0.7094  |  5.8625 |
|  cpu   | torch.float16  |  basic   |          traced         |  0.7094  |  5.6994 |
|  cpu   | torch.float16  |  basic   |          frozen         |  0.7094  |  5.3554 |
|  cpu   | torch.float16  |  basic   |          frozen         |  0.7094  |  5.232  |
|  cpu   | torch.float16  |  basic   | optimized_for_inference |  0.7094  |  4.1322 |
|  cpu   | torch.float16  |  basic   | optimized_for_inference |  0.7094  |  4.1168 |
|  cpu   | torch.float16  |  basic   |         compiled        |  0.7094  |  5.8453 |
|  cpu   | torch.float16  |  basic   |         compiled        |  0.7094  |  6.4078 |
| :----: | :------------: | :------: | :---------------------: | :------: | :-----: |
|  cpu   | torch.float32  |  basic   |           raw           |  0.7094  |  5.4881 |
|  cpu   | torch.float32  |  basic   |         scripted        |  0.7094  |  6.5338 |
|  cpu   | torch.float32  |  basic   |         scripted        |  0.7094  |  5.5762 |
|  cpu   | torch.float32  |  basic   |          traced         |  0.7094  |  6.3419 |
|  cpu   | torch.float32  |  basic   |          traced         |  0.7094  |  6.4318 |
|  cpu   | torch.float32  |  basic   |          frozen         |  0.7094  |  4.8876 |
|  cpu   | torch.float32  |  basic   |          frozen         |  0.7094  |  4.918  |
|  cpu   | torch.float32  |  basic   | optimized_for_inference |  0.7094  |  4.203  |
|  cpu   | torch.float32  |  basic   | optimized_for_inference |  0.7094  |  4.4882 |
|  cpu   | torch.float32  |  basic   |         compiled        |  0.7094  |  5.8761 |
|  cpu   | torch.float32  |  basic   |         compiled        |  0.7094  |  5.6426 |

Based on the results reported in Table 3, execution time does not consistently improve under model optimization. 
Specifically, the '*optimized_for_inference*' method on the CUDA device exhibits higher execution time, whereas on the CPU it achieves the best performance. 
From an accuracy perspective, none of the optimization methods impact accuracy on the CPU. 
In contrast, on the CUDA device, the '*frozen*' optimization method yields the highest accuracy when using the `torch.float16` data type.
When using the `torch.float16` and `torch.bfloat16` data types, a slight reduction in processing time can be observed.

Tab 4) Test-Time Augmentation, using SVHN + CIFAR10 + OxfordIIITPet as dataset
| Device |     Dtype      |         TTA Type        | Model Type | Accuracy | Elapsed |
| :----: | :------------: | :---------------------: | :--------: | :------: | :-----: |
|  cuda  | torch.bfloat16 |          basic          |   frozen   |  0.709   |  0.6433 |
|  cuda  | torch.bfloat16 |          mirror         |   frozen   |  0.7146  |  0.7522 |
|  cuda  | torch.bfloat16 |        translate        |   frozen   |  0.714   |  0.725  |
|  cuda  | torch.bfloat16 | mirroring_and_translate |   frozen   |  0.7191  |  1.1869 |
| :----: | :------------: | :---------------------: | :--------: | :------: | :-----: |
|  cuda  | torch.float16  |          basic          |   frozen   |  0.7094  |  0.6214 |
|  cuda  | torch.float16  |          mirror         |   frozen   |  0.7143  |  0.7244 |
|  cuda  | torch.float16  |        translate        |   frozen   |  0.7133  |  0.7239 |
|  cuda  | torch.float16  | mirroring_and_translate |   frozen   |  0.7199  |  1.1452 |
| :----: | :------------: | :---------------------: | :--------: | :------: | :-----: |
|  cuda  | torch.float32  |          basic          |   frozen   |  0.7094  |  0.6831 |
|  cuda  | torch.float32  |          mirror         |   frozen   |  0.7146  |  0.8592 |
|  cuda  | torch.float32  |        translate        |   frozen   |  0.7136  |  0.7513 |
|  cuda  | torch.float32  | mirroring_and_translate |   frozen   |  0.7197  |  1.5268 |

Based on the results reported in Table 4, several conclusions can be drawn. The most effective TTA methods are '*mirror*' and '*mirroring_and_translate*'. When comparing results after applying TTA, the selected data type has only a minimal impact on accuracy. Overall, the best balance between accuracy and processing time is obtained by using the model with the `torch.float16` data type in combination with the '*mirror*' TTA method.


### Analisis SVHN, fast train, ~1.1 min, run on RTX 4090

The training rate scheduler from: [source](https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench94_muon.py)
```python
class WhitenLrScheduler(LRScheduler):
   #......
   def step(self):
      for group in self.optimizer1.param_groups[:1]:
         group["lr"] = group["initial_lr"] * (1 - self.run_step / self.whiten_bias_train_steps)
      for group in self.optimizer1.param_groups[1:]+self.optimizer2.param_groups:
         group["lr"] = group["initial_lr"] * (1 - self.run_step / self.total_train_steps)
      self.run_step += 1
```
The configurations presented below are adapted from the original [source](https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench94_muon.py), with several modifications introduced to improve accuracy.
```python
EPOCH = 50

W_BATCH_SIZE = 2000
path = "{}/svhn_air_bench".format(DS_PATH)
MEAN_DS, STD_DS = train_in.mean(axis=(0, 1, 2)), train_in.std(axis=(0, 1, 2))
# source: https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench94_muon.py
test_dl  = GpuRowDataLoader(path, test_in,  test_out,  False, MEAN_DS, STD_DS, batch_size=W_BATCH_SIZE, aug=None)
train_dl = GpuRowDataLoader(path, train_in, train_out, True,  MEAN_DS, STD_DS, batch_size=W_BATCH_SIZE, aug=dict(flip=True, translate=2))

model = VGG13(24, train_dl.num_classes)
model = ApplyWhiten2d(model, 3).cuda().to(memory_format=torch.channels_last)
model.compile(mode="max-autotune")

train_images = train_dl.normalize(train_dl.images[:5000])
model.init_whiten(train_images)

criterion = nn.CrossEntropyLoss(reduction="sum", label_smoothing=0.2)
device = torch.device("cuda")

whiten_bias_train_epochs = 5
total_train_steps = np.ceil(EPOCH * len(train_dl)).astype(int)
whiten_bias_train_steps = np.ceil(whiten_bias_train_epochs * len(train_dl)).astype(int)

bias_lr = 0.053
head_lr = 0.97
wd = 2e-8 * W_BATCH_SIZE
train_dl.setWorkMode("basic")
# Create optimizers and learning rate schedulers
filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
norm_biases   = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                 dict(params=norm_biases,         lr=bias_lr, weight_decay=wd/bias_lr),
                 dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True)
optimizer2 = Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True, ns_steps=6)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
        
lr_sheduler1 = WhitenLrScheduler(optimizers, total_train_steps, whiten_bias_train_steps)
lr_shedulers = [lr_sheduler1]
```

Tab 5) Speed training, using SVHN as dataset, run on RTX 4090
| run | epoch | train_acc | val_acc | eval_acc | time_seconds |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
|  0  |   0   |    0.07   |  0.055  |          |    10.938    |
|  0  |   1   |   0.192   |  0.113  |          |    12.634    |
|  0  |   2   |   0.277   |  0.264  |          |    13.641    |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
|  0  |   47  |   0.999   |  0.755  |          |    69.017    |
|  0  |   48  |   0.999   |  0.759  |          |    70.044    |
|  0  |   49  |   0.999   |  0.762  |          |    71.042    |
|  0  |  tta  |           |         |  0.767   |    **75.609**    |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
|  1  |   0   |   0.087   |  0.076  |          |    1.025     |
|  1  |   1   |   0.223   |   0.16  |          |    2.051     |
|  1  |   2   |   0.317   |  0.271  |          |    3.047     |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
|  1  |   47  |   0.999   |  0.752  |          |    47.637    |
|  1  |   48  |   0.999   |  0.758  |          |    48.618    |
|  1  |   49  |   0.999   |  0.761  |          |    49.599    |
|  1  |  tta  |           |         |  0.769   |    **50.081**    |

The results of this experiment were obtained using 50 training epochs and a batch size of 2000. 
The learning rate was set to 0.053 for bias parameters, 0.97 for the head (i.e., the output layer), and 0.24 for convolutional layers. 
Stochastic Gradient Descent (SGD) was applied exclusively to the parameters of the normalization layers, convolutional biases, and the linear output layer, while the Muon optimization method was used solely for the convolutional weights. Best accuracy is **76.9%**, the average training time for the 2 training experiments is **62.845** seconds, device RTX 4090.


### Analisis SVHN, fast learn, ~1.5 min, run on RTX 4090
The configurations presented below are adapted from the original [source](https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench94_muon.py), with several modifications introduced to improve accuracy.
```python
EPOCH = 80

W_BATCH_SIZE = 2000
path = "{}/svhn_air_bench".format(DS_PATH)
MEAN_DS, STD_DS = train_in.mean(axis=(0, 1, 2)), train_in.std(axis=(0, 1, 2))
test_dl  = GpuRowDataLoader(path, test_in,  test_out,  False, MEAN_DS, STD_DS, batch_size=W_BATCH_SIZE, aug=None)
train_dl = GpuRowDataLoader(path, train_in, train_out, True,  MEAN_DS, STD_DS, batch_size=W_BATCH_SIZE, aug=dict(flip=True, translate=2))

model = VGG13(24, train_dl.num_classes)
model = ApplyWhiten2d(model, 3).cuda().to(memory_format=torch.channels_last)
model.compile(mode="max-autotune")

train_images = train_dl.normalize(train_dl.images[:5000])
model.init_whiten(train_images)

criterion = nn.CrossEntropyLoss(reduction="sum", label_smoothing=0.2)
device = torch.device("cuda")

whiten_bias_train_epochs = 5
total_train_steps = np.ceil(EPOCH * len(train_dl)).astype(int)
whiten_bias_train_steps = np.ceil(whiten_bias_train_epochs * len(train_dl)).astype(int)

bias_lr = 0.053
norm_lr = 0.53
head_lr = 0.67
wd = 2e-9 * W_BATCH_SIZE
train_dl.setWorkMode("basic")
# Create optimizers and learning rate schedulers
filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
norm_biases   = [p for p in list(model.parameters())[:-1] if len(p.shape) == 1 and p.requires_grad]
param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                 dict(params=norm_biases,         lr=norm_lr, weight_decay=wd/bias_lr),
                 dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
optimizer1 = torch.optim.SGD(param_configs, momentum=0.1, nesterov=True)
optimizer2 = Muon(filter_params, lr=0.001, momentum=0.6, nesterov=True, ns_steps=3)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
        
lr_scheduler1 = WhitenLrScheduler(optimizers, total_train_steps, whiten_bias_train_steps)
lr_schedulers = [lr_scheduler1]
```


Tab 6) Speed training, using SVHN as dataset, run on RTX 4090
| run | epoch | train_acc | val_acc | eval_acc | time_seconds |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
|  0  |   0   |   0.041   |  0.015  |          |    1.026     |
|  0  |   1   |   0.101   |  0.025  |          |    2.038     |
|  0  |   2   |    0.15   |  0.031  |          |    3.101     |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
|  0  |   77  |   0.648   |  0.499  |          |    79.88     |
|  0  |   78  |   0.648   |  0.498  |          |    80.879    |
|  0  |   79  |   0.648   |  0.499  |          |    81.874    |
|  0  |  tta  |           |         |  **0.507**   |    83.406    |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
|  1  |   0   |    0.05   |  0.011  |          |    1.162     |
|  1  |   1   |   0.107   |  0.026  |          |    2.291     |
|  1  |   2   |   0.152   |  0.093  |          |    3.406     |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
|  1  |   77  |   0.629   |  0.491  |          |    87.866    |
|  1  |   78  |   0.631   |  0.491  |          |    88.951    |
|  1  |   79  |   0.631   |  0.492  |          |    90.033    |
|  1  |  tta  |           |         |  **0.497**   |    90.733    |

The results of this experiment were obtained using 80 training epochs and a batch size of 2000. 
The learning rate was set to **0.053** for bias parameters, **0.67** for the head (i.e., the output layer), **0.53** for normalization layers and **0.001** for convolutional layers. 
Stochastic Gradient Descent (SGD) was applied exclusively to the parameters of the normalization layers, convolutional biases, and the linear output layer, while the Muon optimization method was used solely for the convolutional weights. Best accuracy is **49.7%**, the average training time for the 2 training experiments is **86.72** seconds, device RTX 4090.


### Analisis SVHN, fast learn, ~7.1 min, run on RTX 4060
The configurations presented below are adapted from the original [source](https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench94_muon.py), with several modifications introduced to improve accuracy.
```python
EPOCH = 80

W_BATCH_SIZE = 2000
path = "{}/svhn_air_bench".format(DS_PATH)
MEAN_DS, STD_DS = train_in.mean(axis=(0, 1, 2)), train_in.std(axis=(0, 1, 2))
test_dl  = GpuRowDataLoader(path, test_in,  test_out,  False, MEAN_DS, STD_DS, batch_size=W_BATCH_SIZE, aug=None)
train_dl = GpuRowDataLoader(path, train_in, train_out, True,  MEAN_DS, STD_DS, batch_size=W_BATCH_SIZE, aug=dict(flip=True, translate=2))

model = VGG13(24, train_dl.num_classes)
model = ApplyWhiten2d(model, 3).cuda().to(memory_format=torch.channels_last)
model.compile(mode="max-autotune")

train_images = train_dl.normalize(train_dl.images[:5000])
model.init_whiten(train_images)

criterion = nn.CrossEntropyLoss(reduction="sum", label_smoothing=0.2)
device = torch.device("cuda")

whiten_bias_train_epochs = 50
total_train_steps = np.ceil(EPOCH * len(train_dl)).astype(int)
whiten_bias_train_steps = np.ceil(whiten_bias_train_epochs * len(train_dl)).astype(int)

bias_lr = 0.053
norm_lr = 0.53
head_lr = 0.67
wd = 2e-8 * W_BATCH_SIZE
train_dl.setWorkMode("basic")
# Create optimizers and learning rate schedulers
filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
norm_biases   = [p for p in list(model.parameters())[:-1] if len(p.shape) == 1 and p.requires_grad]
param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                 dict(params=norm_biases,         lr=norm_lr, weight_decay=wd/bias_lr),
                 dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True)
optimizer2 = Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True, ns_steps=3)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
        
lr_scheduler1 = WhitenLrScheduler(optimizers, total_train_steps, whiten_bias_train_steps)
lr_schedulers = [lr_scheduler1]
```

Tab 7) Speed training, using SVHN as dataset, runt on RTX 4060

| run | epoch | train_acc | val_acc | eval_acc | time_seconds |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
|  0  |   0   |   0.051   |  0.024  |          |    26.102    |
|  0  |   1   |   0.156   |  0.141  |          |    30.636    |
|  0  |   2   |   0.258   |   0.22  |          |    35.177    |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
|  0  |   77  |   0.999   |  0.748  |          |    421.02    |
|  0  |   78  |   0.999   |  0.748  |          |   425.497    |
|  0  |   79  |   0.999   |   0.75  |          |   430.087    |
|  0  |  tta  |           |         |  **0.762**   |   443.991    |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
|  1  |   0   |   0.058   |  0.055  |          |    4.643     |
|  1  |   1   |   0.178   |  0.157  |          |    9.247     |
|  1  |   2   |   0.286   |  0.204  |          |    13.896    |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
| :-: | :---: | :-------: | :-----: | :------: | :----------: |
|  1  |   77  |   0.999   |  0.746  |          |   356.805    |
|  1  |   78  |   0.999   |   0.75  |          |   361.257    |
|  1  |   79  |   0.999   |   0.75  |          |   365.707    |
|  1  |  tta  |           |         |  **0.764**   |   367.804    |


The results of this experiment were obtained using 80 training epochs and a batch size of 2000. 
The learning rate was set to 0.053 for bias parameters, 0.67 for the head (i.e., the output layer), 0.53 for normalization layers and 0.24 for convolutional layers. 
Stochastic Gradient Descent (SGD) was applied exclusively to the parameters of the normalization layers, convolutional biases, and the linear output layer, while the Muon optimization method was used solely for the convolutional weights. Best accuracy is **76.4%**, the average training time for the 2 training experiments is **405.9** seconds -> **6.76** min, device RTX 4060.



## Conclusion
The image classification latency is strongly dependent on both the chosen optimization strategy and the execution device. 
Certain optimization modes, such as '*optimized_for_inference*', exhibit higher processing times on CUDA-enabled devices while performing more efficiently on CPUs. 
When using CUDA, both the data type and accuracy configuration have a measurable impact on model performance, as illustrated by the preceding results.

Post-processing techniques yield only marginal accuracy improvements, typically not exceeding 1%. Moreover, post-processing operations that modify brightness, contrast, or saturation tend to degrade performance. 
Among the evaluated methods, mirroring, translation, and the combined mirroring-and-translation approach consistently produced the best results.

A comparison between slower training regimes (approximately 40 minutes with a batch size of 128) and faster training configurations (batch size of 2000) highlights the substantial influence of the learning rate on accuracy. For slower training, optimal performance was achieved with a learning rate of 1×10−4, whereas faster training required a significantly higher learning rate of 0.24. Additionally, the learning rate scheduler plays a critical role in performance optimization: `CosineAnnealingLR`, applied per epoch, proved most effective for slower training, while a gradual per-batch learning rate decay yielded superior results in the fast-training regime.

In summary, deploying classifiers across heterogeneous hardware platforms necessitates comprehensive performance benchmarking on a wide range of devices to determine suitable optimization strategies, data types, and post-processing techniques for each target environment. Effective model training further requires careful hyperparameter fine-tuning; suboptimal results under one configuration do not imply equivalent performance under alternative parameter settings.
For more results see 'main.ipynb'