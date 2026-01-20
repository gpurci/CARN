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
- conducted model training; 
- performed hyperparameter fine-tuning.


## Data augmentation


## Analysis of the best results

### Analisis only SVHN, with next configuration
```python
EPOCH = 300
BATCH_SIZE = 128

model = VGG13(3,  train_ds.num_classes) # train_ds.num_classes get number of classes
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda")
optimizer    = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)
```
Tab 1) Speed test analisis, using only SNHN asa dataset, run time 40 min
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

Tab 2) Speed test analisis, using only SNHN asa dataset
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

Based on the results obtained under these configurations Tab 2, several conclusions can be drawn. The most efficient TTA methods are '*translate*' and '*mirroring_and_translate*'. The '*mixt*' method represents a combination of the following techniques: '*translate*', '*adjust_brightness*', '*adjust_contrast*', '*adjust_saturation*', and '*adjust_hue*'. The results indicate that image rotation and color-range modifications have a negative impact on accuracy. When comparing outcomes after applying TTA, the choice of data type has only a marginal effect on accuracy. Overall, the best trade-off between accuracy and processing time is achieved by using the model with data type `torch.float16` in combination with the '*translate*' TTA method.


### Analisis SVHN + CIFAR10, with next configuration
As a data augmentation strategy, I extended the existing methods by incorporating samples from the CIFAR-10 dataset into the training data. 
For labeling, a counter was used such that each time an image was selected from CIFAR-10, the corresponding counter value was assigned as its label.
```python
EPOCH = 300
BATCH_SIZE = 128

model = VGG13(3,  train_ds.num_classes) # train_ds.num_classes get number of classes
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda")
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-9)
```
Tab 3) Speed test analisis, using SVHN + CIFAR10 asa dataset, run time 40 min
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


Tab 4) Speed test analisis, using SVHN + CIFAR10 asa dataset
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

Based on the results reported in Table 2, several conclusions can be drawn. The most effective TTA methods are '*mirror*' and '*mirroring_and_translate*'. When comparing results after applying TTA, the selected data type has only a minimal impact on accuracy. Overall, the best balance between accuracy and processing time is obtained by using the model with the `torch.float16` data type in combination with the '*mirror*' TTA method.

### Designed an image augmentation pipeline

When creating the `GeneticAlgorithm` object, you can control every stage of the process through configuration dictionaries.

Example:

```python
ttp = GeneticAlgorithm(
    name="test",
    extern_commnad_file="extern_command.cmd",
    metric={"method": "TSP"},
    init_population={"method": "TSP_aleator"},
    fitness={"method": "TSP_f1score"},
    select_parent={
        "select_parent1": {"method": "turneu"},
        "select_parent2": {"method": "turneu_choice"},
    },
    crossover={"method": "mixt"},
    mutate={"method": "mixt"},
    callback="logs/history.csv"
)
```
Each configuration specifies how one component of the genetic algorithm behaves.

---

## ️ Configuration Options

| Parameter | Purpose | Example Values |
|------------|----------|----------------|
| **`metric`** | Defines how the problem is evaluated (distance, time, profit, etc.). | `"TSP"` or `"TTP"` |
| **`init_population`** | Defines how the initial population (the starting solutions) is generated. | `"TSP_aleator"` (random), `"TTP_vecin"` (heuristic) |
| **`fitness`** | Determines how the “quality” of a solution is measured. | `"TSP_f1score"`, `"TSP_norm"`, `"TTP_linear"`, `"TTP_exp"` |
| **`select_parent1`, `select_parent2`** | Define how parents are chosen for reproduction. Two strategies can be used for better diversity. | `"turneu"`, `"turneu_choice"`, `"roata"` |
| **`crossover`** | Defines how two parents are combined into a child (genetic recombination). | `"split"`, `"perm_sim"`, `"mixt"` |
| **`mutate`** | Mutation strategy for introducing randomness and variation in offspring. | `"swap"`, `"inversion"`, `"mixt"` |
| **`callback`** | Path to the CSV file used for logging performance metrics across generations. | `"logs/history.csv"` |
| **`extern_commnad_file`** | Optional file that allows stopping or controlling the algorithm externally. | `"extern_command.cmd"` |

---

##  GA Parameters

Once the algorithm is configured, you can fine-tune its runtime parameters with:

```python
ttp.setParameters(
    GENERATIONS=5000,      # Number of generations (iterations)
    POPULATION_SIZE=1000,  # Number of individuals in each generation
    MUTATION_RATE=0.05,    # Probability of mutation
    CROSSOVER_RATE=0.99,   # Probability of crossover
    SELECT_RATE=0.99,      # Fraction of population used for selection
    ELITE_SIZE=50          # Top individuals preserved automatically each generation
)
```
These control the evolution process — larger populations and more generations improve results but increase runtime.

---

## TTP Generator Module

The file **`ttp_generator.py`** provides the `TTPGenerator` class — a helper used to **load and prepare data** for the  
**Travelling Thief Problem (TTP)** and **visualize routes** generated by the Genetic Algorithm.

This module handles:
- Reading city coordinates and item data from `.csv` files.  
- Computing pairwise distances between all cities.  
- Linking each city with assigned item **profit** and **weight**.  
- Generating visual map images to display routes.

---

### How It Works

The `TTPGenerator` class builds a dataset that your GA can directly use.

```python
from ttp_generator import TTPGenerator

# Initialize with dataset folder path
ttp_generator = TTPGenerator(path="datasets")

# Load city coordinates and item data (tab-separated CSVs)
dataset = ttp_generator("nodes.csv", "items.csv")

# The dataset contains:
# - GENOME_LENGTH: number of cities
# - distance: matrix of pairwise distances
# - coords: list of (x, y) positions
# - item_profit: array of profit values
# - item_weight: array of item weights
```

You can then pass this `dataset` to the main Genetic Algorithm runner:

```python
from genetic_AO_man import GeneticAlgorithm

# Create and configure the Genetic Algorithm instance
ttp = GeneticAlgorithm(
    name="ttp_example",
    metric={"method": "TTP"},
    init_population={"method": "TTP_aleator"},
    fitness={"method": "TTP_linear"},
    select_parent={
        "select_parent1": {"method": "turneu"},
        "select_parent2": {"method": "roata"},
    },
    crossover={"method": "mixt"},
    mutate={"method": "mixt"},
    callback="logs/history.csv"
)

# Define algorithm parameters
ttp.setParameters(
    GENERATIONS=1000,
    POPULATION_SIZE=500,
    MUTATION_RATE=0.05,
    CROSSOVER_RATE=0.95,
    SELECT_RATE=0.9,
    ELITE_SIZE=50
)

# Run the algorithm using the TTP dataset
ttp(dataset)
```
This will execute a **TTP optimization** where each generation improves the trade-off between **travel distance** and **total profit collected**.  
The algorithm uses evolutionary principles — **selection, crossover, and mutation** — to evolve increasingly efficient solutions.  
During the process, results are continuously logged and can later be analyzed or visualized.

---

###  Outputs

When you run the algorithm:
- Progress is logged in **`logs/history.csv`**  
- Console output shows each generation’s best fitness and score  
- (Optional) Route visualization displays the best found path on the map  

**Example console output:**
```commandline
Running Genetic Algorithm (TTP)
Generation 50: best_profit=245.7, distance=312.4, fitness=0.89
Generation 51: best_profit=249.3, distance=308.1, fitness=0.91
...
Optimization completed after 1000 generations.
```
###      Adjusting Parameters for TTP

You can fine-tune the Genetic Algorithm parameters in `genetic_AO_man.py` or directly in your own script to control runtime and performance:

```python
ttp.setParameters(
    GENERATIONS=2000,       # Number of generations (iterations)
    POPULATION_SIZE=800,    # Number of individuals per generation
    MUTATION_RATE=0.05,     # Probability of mutation
    CROSSOVER_RATE=0.95,    # Probability of crossover
    ELITE_SIZE=50           # Number of top individuals kept each generation
)
```

#### Parameter Meaning

Each parameter directly influences how your Genetic Algorithm behaves and evolves over time.  
Understanding their effects helps you fine-tune performance and balance exploration with exploitation.

| **Parameter** | **Description** | **Effect on Algorithm** | **Recommended Range** |
|----------------|-----------------|--------------------------|------------------------|
| **GENERATIONS** | Total number of iterations the algorithm will run. | Higher values allow for deeper evolution but increase runtime. | 500 – 10,000 |
| **POPULATION_SIZE** | Number of individuals in each generation. | Larger values improve diversity and stability but increase computational cost. | 100 – 1,000+ |
| **MUTATION_RATE** | Probability of applying random changes to individuals. | High values improve exploration but may reduce convergence speed. | 0.01 – 0.1 |
| **CROSSOVER_RATE** | Probability that two selected parents will crossover to create offspring. | Encourages exploitation by combining good traits; too low may reduce diversity. | 0.8 – 0.99 |
| **ELITE_SIZE** | Number of top individuals preserved unchanged into the next generation. | Maintains high-quality solutions; too large can reduce genetic diversity. | 10 – 50 |
 **Tip:** If your algorithm converges too quickly (fitness stops improving early), try:
- Increasing the **mutation rate**
- Reducing the **elite size**
- Increasing **population size**

If it’s too random and unstable, try:
- Lowering the **mutation rate**
- Increasing the **elite size**
- Reducing **population size**

---

###  Example Parameter Tuning

Here’s a quick Python snippet showing how to experiment with different configurations to find the best-performing setup:

```python
# Example parameter sweep
for population in [200, 400, 800]:
    for mutation in [0.02, 0.05, 0.1]:
        print(f"Running test: POPULATION={population}, MUTATION={mutation}")
        ttp.setParameters(
            GENERATIONS=1000,
            POPULATION_SIZE=population,
            MUTATION_RATE=mutation,
            CROSSOVER_RATE=0.95,
            ELITE_SIZE=30
        )
        ttp(dataset)
```




The CSV log can be plotted to analyze convergence trends, showing how fitness improves with each generation.

---

### Interpreting the Results

- **Fitness** measures how well a given solution balances profit and distance.  
- **Score** typically reflects total distance or another optimization metric depending on the selected fitness method.  
- **Profit and distance** trade-offs evolve over generations — you can compare these to evaluate different configurations.

## Practical Tips

- Start with small values for `POPULATION_SIZE` and `GENERATIONS` to validate your setup quickly.  
- Use the `"mixt"` mode for **crossover** and **mutation** to automatically combine multiple strategies.  
- Try different **selection methods** (`turneu`, `roata`, `choice`) to adjust diversity and convergence speed.  
- To view all available options and methods, simply run:
  ```python
  GeneticAlgorithm().help()
    ```
- To stop the algorithm gracefully mid-run, edit your external command file (`extern_command.cmd`) and set:

    ```yaml
    stop: true
    ```
  This command allows you to safely interrupt long-running optimizations without losing progress or corrupting output logs.  
The algorithm periodically checks this file during execution, so the stop command is detected quickly and handled gracefully.

---

##  Algorithm Workflow
```
Population → Fitness Evaluation → Selection → Crossover → Mutation → New Generation
                   ↑                                                      ↓
              Metrics & Logging   ←  Elitism (best individuals preserved)
```


This cycle represents the **core loop** of the Genetic Algorithm.  
Each stage plays a specific and essential role in the evolutionary process:

- **Population** → The current set of individuals (possible solutions) in the algorithm.  
- **Fitness Evaluation** → Each individual is evaluated based on the chosen objective or metric (for example, total route distance or total profit).  
- **Selection** → The best individuals are selected as parents to produce the next generation.  
- **Crossover** → Parents exchange parts of their genetic material to create new offspring (solutions).  
- **Mutation** → Small random changes are introduced to maintain diversity and prevent stagnation.  
- **New Generation** → The new set of individuals created after crossover and mutation replaces the previous population.  
- **Elitism** → A few of the top-performing individuals are carried forward unchanged to ensure the best traits are preserved.

The algorithm continues through this cycle until:
- The **maximum number of generations** is reached, **or**
- An **external stop command** (`stop: true`) is detected in the `extern_command.cmd` file.

---
###  Key Takeaways

Optimizing a Genetic Algorithm isn’t just about finding “one best” configuration — it’s about balancing parameters to suit your specific problem and dataset.  
Here are the main ideas to keep in mind:

- **Generations** → Define how long the algorithm evolves. More generations give better refinement but increase runtime.  
- **Population size** → Controls diversity. Larger populations explore more possibilities but take longer per iteration.  
- **Mutation rate** → Encourages exploration. Too high makes results chaotic; too low may trap you in local minima.  
- **Crossover rate** → Governs exploitation. Higher values combine good solutions efficiently; lower values maintain more diversity.  
- **Elitism** → Protects the best individuals. A small elite size helps preserve high-quality solutions while still allowing evolution.

To find the right balance:
1. Start with default parameters.  
2. Observe the convergence trend in `logs/history.csv`.  
3. Gradually adjust **mutation**, **elitism**, and **population size** to improve stability and convergence.  

---

###  Example Tuning Workflow

1. **Run baseline tests** with default parameters to ensure stability.  
2. **Increase population size** to see if diversity improves results.  
3. **Adjust mutation rate** (0.01–0.1) to fine-tune exploration.  
4. **Use multiple selection methods** (e.g., `"turneu"` and `"roata"`) for better variety.  
5. **Compare runs** visually using fitness plots from your CSV logs.  

Through systematic experimentation, you’ll quickly learn which configurations best suit your optimization problem.

---
##  Summary

This framework provides a **modular and flexible system** for experimenting with Genetic Algorithms in Python.  
It separates the evolutionary process into independent, replaceable modules, making it perfect for experimentation and learning.

### Key Features:
- Built-in support for **TSP (Travelling Salesman Problem)** and **TTP (Travelling Thief Problem)**  
- Modular architecture — each algorithmic phase (metrics, initialization, selection, crossover, mutation) is customizable  
- Multiple strategies for selection, mutation, and crossover  
- Automatic CSV logging for performance tracking  
- Optional route visualization and safe external stop control  

---

### Final Thoughts

This project provides a **powerful yet easy-to-understand framework** for implementing and experimenting with Genetic Algorithms in Python.  
It focuses on clarity, modularity, and adaptability — making it suitable for **students, researchers, and developers** alike.

Whether you're studying evolutionary computation concepts, optimizing real-world problems, or designing your own operators, this framework gives you full control over every component of the algorithm.

---