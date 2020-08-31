---
date: 2020-06-21 18:39:28
layout: post
title: Understanding Metric Learning, Chapter2
subtitle: Research
# description: Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
math: true
image: https://res.cloudinary.com/dm7h7e8xj/image/upload/v1559824306/theme13_dshbqx.jpg
optimized_image: https://res.cloudinary.com/dm7h7e8xj/image/upload/c_scale,w_380/v1559824306/theme13_dshbqx.jpg
category: Machine Learning
tags:
    - Metric Learning
    - Triplet
    - Pairwise
author: pyy0715
---

# Understanding Metric Learning, Chapter2

[지난 포스트](https://pyy0715.github.io/Metric_Learning1/)에 이어서 Metric Learning에 관한 2번째 포스트입니다.
다시 한번 살펴보면 Triplet loss의 학습은 목적은 아래와 같습니다.

- 임베딩 공간 내에서 Anchor는 **Positive Sample**과의 거리가 가까워야 합니다.
- 임베딩 공간 내에서 Anchor는 **Negative Sample**과의 거리는 멀어져야 합니다.

$$\mathcal{L} = max(d(a, p) - d(a, n) + margin, 0)$$

그러나 `easy triplet`의 경우 loss가 0이 되어서 학습에 영향을 끼치지 않는다고 하였습니다.
따라서 학습 효율성을 높이기 위해 좀 더 스마트한 Sampling 기법들이 필요하게 되었습니다.

## Strategies for Triplet Selection

### Offline Triplet Mining

Triplet loss를 만드는 첫번째 방법은 Train 데이터에서 모든 임베딩을 계산한 후에, `semi-hard`, `hard` 를 만족하는 쌍을 찾는 것입니다.
이러한 방법은 데이터에서 모든 쌍을 구해야 하기 때문에 계산적인 측면에서 비효율적입니다.
또한 새로운 데이터가 들어올 시, 업데이트를 정기적으로 해주어야 한다는 단점도 있습니다.


### Online Triplet Mining

위의 문제점을 해결하기 위해  주어진 batch 단위 내에서 임베딩을 계산하고, triplet 쌍을 찾도록 하였습니다.
하지만 단순히 sampling한 batch의 triplet 쌍은 유효하지 않습니다.
(예를들어 batch 내에서 anchor의 positive나 negative sample이 없을 수도 있기 때문입니다.)

따라서 batch를 잘 구성하는 것이 중요하였기 때문에 [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)에서 소개된 2가지의 방법을 살펴보겠습니다.
먼저 기본적인 가정으로 batch의 size는 $B=PK$로 구성하였는데, 여기서 $K$개의 이미지 당, $P$명의 서로 다른 사람으로 이루어져 있습니다.
예를들면 $K=4$일때,  $P$명의 사람은 4개의 이미지를 가지고 있어야 합니다.


### (1) Batch All

batch 내 유효한 triplet 쌍을 선택하고, `semi-hard`, `hard` 를 만족하는 모든 triplet의 평균 loss를 구합니다.
FaceNet에서도 batch 내에서 anchor의 모든 positive sample를 구하면서 negative sample의 경우
아래 수식을 만족하는 semi-hard 방식으로 negative sample을 샘플링했습니다.

$$\left\|f\left(x_{i}^{a}\right)-  
f\left(x_{i}^{p}\right)\right\|_{2}^{2}<\left\|f\left(x_{i}^{a}\right)-f\left(x_{i}^{n}\right)\right\|_{2}^{2}$$

여기서 `easy triplet` 은 고려하지 않습니다.
전체 triplet 사이즈는 $PK$ * $(K−1)$ * $(PK−K)$로 구성됩니다.

> $PK$가 anchor
>
> $K-1$는 anchor의 가능한 positive sample 수
>
> $PK-K$는 anchor의 가능한 negatvie sample 수

코드는 아래와 같습니다.

```python
def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss[triplet_loss < 0] = 0

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
    print(triplet_loss, fraction_positive_triplets)

    return triplet_loss, fraction_positive_triplets
```

### (2) Batch Hard

batch 내의 각 anchor에서 hardest positive, hardest negative를 구합니다.

$$x_{h p}=\underset{x: C(x)=C\left(x_{a}\right)}{\operatorname{argmax}} d\left(f\left(x_{a}\right), f(x)\right)$$

$$x_{h n}=\underset{x: C(x) \neq C\left(x_{a}\right)}{\operatorname{argmin}} d\left(f\left(x_{a}\right), f(x)\right)$$

이는 쉬운 문제보다는 어려운 문제를 풀게 하면서 학습을 시키는 것입니다.
모델이 어려워 할 만한 Anchor와 가장 멀리 있는 Postivie, 가장 가까이 있는 Negative로 학습합니다.
**즉, hard point로 학습을 시키자는 것이 핵심입니다.**

이 떄의 triplet 사이즈는 $B=PK$입니다.
실험적으로 위의 방법보다 batch hard 방법이 성능적으로 더 우수하다고 합니다.
하지만 데이터에 전적으로 의존하기 때문에 여러가지 실험이 필요합니다.

```python
def batch_hard_triplet_loss(labels, embeddings, margin, squared=False, device='cpu'):
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels, device).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl[tl < 0] = 0
    triplet_loss = tl.mean()

    return triplet_loss
```

## Next

이번 포스트에는 triplet loss를 효과적으로 학습시키기 위해 2가지의 sampling 방법을 살펴보았습니다.
머신러닝에서도 일반화를 잘 시키기 위해서 k-fold의 방법을 사용하는 것처럼 모델을 잘 만드는 것만큼이나 sampling을 잘 하는 것도 중요하다고 느껴졌습니다.
다음 포스트에는 metric learning이 추천의 분야에서는 어떻게 사용되고 있는지 Netflix의 발표자료를 통해서 알아보겠습니다.

## Reference

[KevinMusgrave/pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

[Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss#a-better-implementation-with-online-triplet-mining)
