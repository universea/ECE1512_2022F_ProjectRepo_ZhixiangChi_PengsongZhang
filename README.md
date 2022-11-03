# ECE1512_2022F_ProjectRepo_ZhixiangChi_PengsongZhang



Zhixiang Chi, Pengsong Zhang



#### Task 1:

The code for task 1 can be found in Task1.ipynb. The trained models are also included in the corresponding folders.

#### Task 2:

Download pretrained weights: 'resnetv2_50_a1h-000cdf49.pth', 'mobilenetv2_100_ra-b33bc2c4.pth', 'resnet18-5c106cde.pth', and place them in the folder './pretrained_models'.

To train student model from scratch:

```python
CUDA_VISIBLE_DEVICES=0 python Task_2_KD_system.py --student_scratch
```

To fine-tune the student model:

```python
CUDA_VISIBLE_DEVICES=0 python Task_2_KD_system.py --student_finetune
```

To train the student network by distilling knowledge from teacher:

```python
CUDA_VISIBLE_DEVICES=0 python Task_2_KD_system.py --student_KD
```

To use ResNet-18 model as an assistant to distill the student model:

```python
CUDA_VISIBLE_DEVICES=0 python Task_2_KD_system.py --use_assistant
```

To use both teacher and assistant during the knowledge distillation:

```python
CUDA_VISIBLE_DEVICES=0 python Task_2_KD_system.py --use_both_teachers
```

To run all the above experiments:

```python
CUDA_VISIBLE_DEVICES=0 python Task_2_KD_system.py --complete
```

To test the sensitivity of the temperatures for both student and assistant network:

```python
CUDA_VISIBLE_DEVICES=0 python Task_2_KD_system.py --student_temp_values --assistant_temp_values
```



The logs for the complete experiments with 30 and 150 epochs are provided as: 

