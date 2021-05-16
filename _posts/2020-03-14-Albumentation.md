---
date: 2020-03-14 18:39:28
layout: post
title: Albumentation Library 소개
subtitle: Tutorial
# description: Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
math: true
image: https://res.cloudinary.com/dm7h7e8xj/image/upload/v1559822137/theme11_vei7iw.jpg
optimized_image: https://res.cloudinary.com/dm7h7e8xj/image/upload/c_scale,w_380/v1559822137/theme11_vei7iw.jpg
category: Vision
tags:
    - Albumentation
    - Augmentation
    - CNN
author: pyy0715
---

# What is Augmentation?

오늘은 최근 블로그에 쓰고 있던 주제와는 다른 글을 가져오게 되었습니다.
최근 Kaggle의 [Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19) 대회에 참가하면서,
오랜만에 이미지에 관한 기법들이나 모델들을 공부하게 되었습니다.
Kaggle의 이미지 대회에서는 모델의 성능을 끌어올리기 위해, Augmentation 기법들을 사용하게 됩니다.

Augmentation이란 머신러닝의 고질적인 문제인 Overfitting을 방지하기 위해, 기존의 데이터에 약간의 Noise를 주는 것입니다.
약간의 Noise를 부여함으로써, Overfitting을 방지하고 모델의 성능을 끌어올릴 수 있습니다.

아래의 예제처럼, 고양이를 분류하는 문제에서 다른 각도의 고양이들을 같이 학습하면 좀 더 좋은 성능을 낼 수 있지 않을까요?

<img src='https://www.kdnuggets.com/wp-content/uploads/cats-data-augmentation.jpg'>

저 역시 대회를 진행하면서, 모델의 성능을 올리기 위해 `Albumentation` 이라는 이미지 Augmentation Library를 알게 되었고, 사용하여 좋은 결과를 낼 수 있었습니다.
`Albumentation` 은 다른 Library들과도 사용하기 쉬우며, 빠른 이미지 Augmentation을 도와줍니다.

이번 글에서는 Albumentation Library 사용법과 함께, AugMix라는 최신 기법까지 알아보겠습니다.
튜토리얼은 [공식 Github](https://github.com/albumentations-team/albumentations)을 기반으로 작성되었고, 사용한 데이터는 대회 데이터의 일부를 사용하였습니다.
아래의 모든 코드는 [Jupyter Notebook](https://nbviewer.jupyter.org/gist/pyy0715/ab99bcee0f535cb7b75419ae2e8be17f)에서 확인 가능합니다.

# Albumentation Tutorial Notebook

## Import Module
필요한 모듈을 불러옵니다.

```python
import pandas as pd
import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import albumentations
```

## Original Data
Augmentation을 적용하지 않은 초기 데이터입니다.

```python
nrow, ncol = 1, 5

fig, axs = plt.subplots(nrow, ncol, figsize=(20, 10))
axs = axs.flatten()

for i, ax in enumerate(axs):
    img = image[i]
    ax.imshow(img)
    ax.set_title(f'label: Original')
    ax.axis('off')
plt.tight_layout()
```
![output_10_0](https://user-images.githubusercontent.com/47301926/76653209-92819400-65ab-11ea-9e99-61145039a325.png)

## Blur

> *Blur the input image using a random-sized kernel.*

임의 크기의 커널을 사용하여 이미지를 흐리게 만듭니다.

```python
import albumentations as A

aug = A.Blur(p=0.5)

nrow, ncol = 5, 2
fig, axs = plt.subplots(nrow, ncol, figsize=(20, 10))

for n in range(nrow):
    img = image[n]
    aug_image = aug(image=img)['image']

    axs[n,0].imshow(img)
    axs[n,0].set_title(f'label: Original')
    axs[n,0].axis('off')

    axs[n,1].imshow(aug_image)
    axs[n,1].set_title(f'label: Blur Image')
    axs[n,1].axis('off')

plt.tight_layout()
```
![output_13_0](https://user-images.githubusercontent.com/47301926/76653210-92819400-65ab-11ea-9874-c025988d1c99.png)

## Noise

> *Apply gaussian noise to the input image.*

이미지에 noise를 더하여, 좀 더 robust한 결과를 만들도록 합니다.

```python
import albumentations as A

aug = A.GaussNoise(var_limit=5. / 255., p=1.0)

nrow, ncol = 5, 2
fig, axs = plt.subplots(nrow, ncol, figsize=(20, 10))

for n in range(nrow):
    img = image[n]
    aug_image = aug(image=img)['image']

    axs[n,0].imshow(img)
    axs[n,0].set_title(f'label: Original')
    axs[n,0].axis('off')

    axs[n,1].imshow(aug_image)
    axs[n,1].set_title(f'label: Gauss Noise')
    axs[n,1].axis('off')

plt.tight_layout()
```

![output_16_0](https://user-images.githubusercontent.com/47301926/76653213-931a2a80-65ab-11ea-91e1-b28dc17fa15b.png)


## Cut Out

> *Course Drop out of the square regions in the image.*

이미지에서 Dropout을 적용한다고 생각하시면 될 꺼 같습니다.

```python
import albumentations as A

aug = A.Cutout(num_holes=8,  max_h_size=20, max_w_size=20, p=1.0)

nrow, ncol = 5, 2
fig, axs = plt.subplots(nrow, ncol, figsize=(20, 10))

for n in range(nrow):
    img = image[n]
    aug_image = aug(image=img)['image']

    axs[n,0].imshow(img)
    axs[n,0].set_title(f'label: Original')
    axs[n,0].axis('off')

    axs[n,1].imshow(aug_image)
    axs[n,1].set_title(f'label: Cut Out')
    axs[n,1].axis('off')

plt.tight_layout()
```

![output_19_0](https://user-images.githubusercontent.com/47301926/76653199-901f3a00-65ab-11ea-8fc5-e306fcb33861.png)


## Brightness, Contrast

> *Randomly change brightness and contrast of the input image.*

이미지의 밝기와 대비를 임의로 변경합니다.


```python
import albumentations as A

aug = A.RandomBrightnessContrast(p=1.0)

nrow, ncol = 5, 2
fig, axs = plt.subplots(nrow, ncol, figsize=(20, 10))

for n in range(nrow):
    img = image[n]
    aug_image = aug(image=img)['image']

    axs[n,0].imshow(img)
    axs[n,0].set_title(f'label: Original')
    axs[n,0].axis('off')

    axs[n,1].imshow(aug_image)
    axs[n,1].set_title(f'label: RandomBrightnessContrast')
    axs[n,1].axis('off')

plt.tight_layout()
```

![output_22_0](https://user-images.githubusercontent.com/47301926/76653205-91506700-65ab-11ea-8873-fa86b688e701.png)


## Scale, Rotate

> *Randomly apply affine transforms: translate, scale and rotate the input*

이미지의 크기나 회전을 임의로 변형시킵니다.


```python
import albumentations as A

aug = A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=30,
                p=1.0)

nrow, ncol = 5, 2
fig, axs = plt.subplots(nrow, ncol, figsize=(20, 10))

for n in range(nrow):
    img = image[n]
    aug_image = aug(image=img)['image']

    axs[n,0].imshow(img)
    axs[n,0].set_title(f'label: Original')
    axs[n,0].axis('off')

    axs[n,1].imshow(aug_image)
    axs[n,1].set_title(f'label: ShiftSclaeRotate')
    axs[n,1].axis('off')

plt.tight_layout()
```

![output_25_0](https://user-images.githubusercontent.com/47301926/76653206-91e8fd80-65ab-11ea-8ef9-692a6d215afe.png)


## Affine

> *Place a regular grid of points on the input and randomly move the neighbourhood of these point around via affine transformations.*

이미지의 격자 내에서 점의 주변을 임의로 이동시킵니다.


```python
import albumentations as A

aug = A.IAAPiecewiseAffine(p=1.0)

nrow, ncol = 5, 2
fig, axs = plt.subplots(nrow, ncol, figsize=(20, 10))

for n in range(nrow):
    img = image[n]
    aug_image = aug(image=img)['image']

    axs[n,0].imshow(img)
    axs[n,0].set_title(f'label: Original')
    axs[n,0].axis('off')

    axs[n,1].imshow(aug_image)
    axs[n,1].set_title(f'label: Affine')
    axs[n,1].axis('off')

plt.tight_layout()
```

![output_28_0](https://user-images.githubusercontent.com/47301926/76653208-91e8fd80-65ab-11ea-9f3d-75cd201afc9d.png)

총 6가지의 대표적인 기법들을 소개했습니다.
이 외에도 많은 Augmentation 기법들이 있으며, 더 자세히는 [공식 Github](https://github.com/albumentations-team/albumentations)를 참고하시는게 좋습니다.

# AugMix

위에서 소개한 Alubumentiation을 기반으로, AugMix가 어떤것인지 알아보겠습니다.
AugMix는 2019년 12월 구글에서 발표한 논문으로, train 과 test sample들의 distribution이 달라 모델의 정확도가 떨어지는 경우 robustness와 uncertainty measure를 크게 향상시킬 수 있는 Data Processing 기법입니다.

> [Paper](https://arxiv.org/abs/1912.02781)
>
> [Official implementation](https://github.com/google-research/augmix)


`CutOut` - 이미지의 임의의 부분을 제거

`MixUp` - 이미지 간 확률적으로 두 이미지를 섞습니다.

`CutMix` - CutOut + MixUp

**`AugMix` - 위에서 소개한 Alubumentiation 기법들을 섞습니다.**

먼저 아래의 사진 4장을 보면서, 대략적인 감을 잡으실 수 있을겁니다.

<img src='https://storage.googleapis.com/groundai-web-prod/media/users/user_135639/project_400799/images/x1.png'>

<img src='https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F448347%2Fe9d3259d5b0ef3dadba0238bf3901f1e%2F2020-02-10%2013.50.53.png?generation=1581310282118474&alt=media'>


CIFAR-10 실험결과, AugMix는 모든 모델에서 다른 기법들보다 우수하게 성적을 냈다고 합니다.
그럼 코드를 살펴보고, 이미지에 어떻게 적용되는지 살펴보겠습니다.

```python
from albumentations.core.transforms_interface import ImageOnlyTransform

class AugMix(ImageOnlyTransform):
    """Augmentations mix to Improve Robustness and Uncertainty.
    Args:
        image (np.ndarray): Raw input image of shape (h, w, c)
        severity (int): Severity of underlying augmentation operators.
        width (int): Width of augmentation chain
        depth (int): Depth of augmentation chain. -1 enables stochastic depth uniformly
          from [1, 3]
        alpha (float): Probability coefficient for Beta and Dirichlet distributions.
        augmentations (list of augmentations): Augmentations that need to mix and perform.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, width=2, depth=2, alpha=0.5, augmentations=[HorizontalFlip()], always_apply=False, p=0.5):
        super(AugMix, self).__init__(always_apply, p)
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.augmentations = augmentations
        self.ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        self.m = np.float32(np.random.beta(self.alpha, self.alpha))

    def apply_op(self, image, op):
        image = op(image=image)["image"]
        return image

    def apply(self, img, **params):
        mix = np.zeros_like(img)
        for i in range(self.width):
            image_aug = img.copy()

            for _ in range(self.depth):
                op = np.random.choice(self.augmentations)
                image_aug = self.apply_op(image_aug, op)

            mix = np.add(mix, self.ws[i] * image_aug, out=mix, casting="unsafe")

        mixed = (1 - self.m) * img + self.m * mix
        if img.dtype in ["uint8", "uint16", "uint32", "uint64"]:
            mixed = np.clip((mixed), 0, 255).astype(np.uint8)
        return mixed

    def get_transform_init_args_names(self):
        return ("width", "depth", "alpha")
```

AugMix는 아래와 같이 여러가지 기법들을 동시에 적용할 수 있으며, 확률값으로 조정이 가능합니다.
또한 Albumentation의 `OneOf`를 사용하여, 기법들 중에서 하나의 기법만 선택할 수 있습니다.


```python
augs = [A.HorizontalFlip(always_apply=True),
        A.Blur(always_apply=True),
        OneOf(
        [A.ShiftScaleRotate(always_apply=True),
        A.GaussNoise(always_apply=True)]
        ),
        A.Cutout(always_apply=True),
        A.IAAPiecewiseAffine(always_apply=True)]

transforms_train = albumentations.Compose([
    AugMix(width=3, depth=2, alpha=.2, p=1., augmentations=augs),
])
```

![image](https://user-images.githubusercontent.com/47301926/76655952-df686900-65b1-11ea-98d6-de92ac2822ba.png)


# Reference

[bengali-albumentations-data-augmentation-tutorial](https://www.kaggle.com/corochann/bengali-albumentations-data-augmentation-tutorial)

[augmix-albumentations-works-with-albu-augs](https://www.kaggle.com/monuwio/augmix-albumentations-works-with-albu-augs)
