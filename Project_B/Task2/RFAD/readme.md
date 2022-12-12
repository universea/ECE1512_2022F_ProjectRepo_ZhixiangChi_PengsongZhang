# Efficient Dataset Distillation using Random Feature Approximation

Code is borrowed from the occifial implementation of the NeurIPS paper ["Efficient Dataset Distillation using Random Feature Approximation"](https://arxiv.org/abs/2210.12067)

To run generate a distilled set on cifar10, 10 samples per class, platt loss with label learning, for example:

```python run_distillation.py --dataset cifar10 --save_path path/to/directory/ --samples_per_class 10 --platt --learn_labels ```

To run generate a distilled set on MNIST, 10 samples per class, platt loss with label learning, for example:

```python run_distillation.py --dataset MNIST --save_path path/to/directory/ --samples_per_class 10 --platt --learn_labels ```
