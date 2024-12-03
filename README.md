# ME455_TermProject
2024 fall ME455 TermProject

## Instruction
Please download the following data:

[checkpoint](https://drive.google.com/)

[dataset](https://drive.google.com/)


The folder should have the following structure:
```
2024_ME455
   ├── student_dataset
       ├── student_test
           ├── current_image
           └── past_image
       └── train
           ├── current_image
           └── past_image
   ├── models
   ├── ckpts
       ├── depth.pth
       ├── encoder.pth
       ├── segmentation.pth
       └── yolov7_cityscapes.pt
   ├── utils
   ├── val_sanity.npy
   └── ME455_TP_2024_Student.ipynb  
 ```

## Model names
Semantic Segmentation: `model_ss`  
Depth Estimation: `depth_encoder`, `depth_decoder`  
Object Detection: `model_od`  