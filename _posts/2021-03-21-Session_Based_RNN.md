---
date: 2021-03-21 18:39:28
layout: post
title: Session-Based Recommendations with Recurrent Neural Networks 리뷰
subtitle: Paper Review
# description: Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
math: true
image: https://res.cloudinary.com/dm7h7e8xj/image/upload/v1559822138/theme9_v273a9.jpg
optimized_image: https://res.cloudinary.com/dm7h7e8xj/image/upload/c_scale,w_380/v1559822138/theme9_v273a9.jpg
category: Recommender System
tags:
  - Session-Based
  - GRU4Rec
author: pyy0715
---

# Session-Based Recommendations with Recurrent Neural Networks

> [Paper](https://arxiv.org/pdf/1511.06939.pdf)

# ABSTRACT

Real Recommender System은 Neflix처럼 Long User Histories를 가지는 경우가 많이 없기 때문에 Short Session-Based를 기반으로 추천이 이루어짐

# Introduction

지금까지 추천시스템에서 사용되었던 기본적인 방법들은 아래와 같습니다.

- `Factor Models`

  Matrix Completion문제로 유저와 아이템의 latent vector를 구해 내적값으로 missing value를 채움
  **Session단위에서는 사용자의 Profile을 이용하지 않기 때문에 접근이 어려움**

- `Neighborhood Methods`

  Session 내 유저 또는 아이템간의 유사성을 계산하는 것을 목적으로 합니다.
  따라서 Session단위 접근법에서는 광범위하게 사용될 수 있다.

E-commerce의 추천시스템은 왜 Session단위로 이루어져야 하는가?

- 한 사용자의 모든 기록을 추적할 수는 없다. (사생활 보호, cookie 데이터의 신뢰성 등)
- 같은 사용자의 연속적인 Session은 독립적으로 다루어야 합니다.
  small e-commerce site에서 사용자의 세션은 1~2개밖에 존재하지 않음

결과적으로 E-commerce의 추천시스템은 상대적으로 간단하게 아이템 간의 유사성을 기반으로 하게 되었다. 이러한 방법은 효과적이지만 결국 사용자의 과거 클릭정보를 무시하게 됩니다.

따라서 저자는 RNN을 이용하여 Session 단위의 정교한 추천 시스템을 만드는 것이 목적입니다. 위의 `Neighborhood Methods`에 기반하여 유저가 처음 클릭한 아이템으로 그 다음 아이템을 예측할 수 있는 모델, 즉 사용자가 관심을 가질 수 있는 상위 항목을 만들려고 합니다.

논문에서 주의깊게 바라보아야 할 점은 아래의 3가지입니다.

- **Sequential한 데이터에서 sparsity를 어떻게 해결하였는지?**
- **추천 Task를 위해 새로운 Ranking Loss는 무엇을 사용하였는지?**
- **click-stream 데이터 또한 상당히 크기 때문에 훈련 시간과 확장성을 어떻게 고려하였는지?**

# 2 Related Work

## 2.1 Session-Based Recommendation

기존 방법들의 경우 사용자의 마지막 클릭 아이템을 기준으로 유사한 아이템을 찾기 때문에,

과거 클릭 정보를 무시하게 되는 문제점이 있다.

이러한 문제점을 해결하기 위해 노력한 2가지의 방법을 소개합니다.

- **MDP(Markov Decision Process)**

  MDP방법을 이용하여 아이템 간 전이확률을 기반으로 간단하게 다음 추천, Action이 무엇인지 계산할 수 있다.

  하지만 빠르게 아이템의 수가 많아지기 때문에, 즉 유저가 선택할 수 있는 폭이 넓어지기 때문에 MDP기반의 접근방법만으로는 state space를 점점 관리할 수 없게 됩니다.

- **GFF(General Factorization Framework)**

  session은 event들의 합으로 표현하며, 아이템에 대해서 두 종류의 latent vector를 사용합니다. 하나는 `the item itself` 와 다른 하나는 `the item as part of a session`을 사용합니다.

  따라서 session은 `the average of the feature vectors of part-of-a-session item`으로 표현될 수 있습니다. 그러나 이러한 방법은 session내에서 순서를 고려하지 않습니다.

## 2.2 Deep Learning In Recommendation

유저와 아이템의 Interaction을 바탕으로 우수한 성능을 내는 방법으로 Restricted Boltzmann Machines (RBM) for Collaborative Filtering이 있습니다. Interaction 정보가 충분하지 않은 경우에 특히 유용합니다.

Deep Model들은 음악이나 이미지같이 구조화되지 않은 컨텐츠에서 feature를 추출하기 위해 사용되어졌으며, 전통적인 CF들과 함께 사용되어져왔습니다.

# 3 Recommendation with RNNs

GRU는 Update Gate와 Reset Gate를 활용하여, 기존 RNN의 문제점인 Vanishing Gradient Problem을 해결하려고 하였습니다.

GRU에 대한 개념과 이미지는 [https://d2l.ai/chapter_recurrent-modern/gru.html](https://d2l.ai/chapter_recurrent-modern/gru.html) 를 참고하였습니다.

## Reset Gate and Update Gate

![Untitled](https://github.com/thiagorossener/jekflix-template/assets/47301926/05ecf954-8c06-4eb5-8f99-acc22825f3a3)

- Reset Gate(capture short-term dependencies)

  controls how much of the previous state we might still want to remember.

  $$
  \begin{split}\begin{aligned}
  \mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xr}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hr}} + \mathbf{b}_\textrm{r}),\\
  \end{aligned}\end{split}
  $$

- Update Gate(capture long-term dependencies)

  controls how much of the new state is just a copy of the old one

  $$
  \mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xz}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hz}} + \mathbf{b}_\textrm{z})
  $$

## Candidate Hidden State

![hidden state](https://github.com/thiagorossener/jekflix-template/assets/47301926/fc9b6c71-3353-4fb9-bfb8-72622d9bacd9)

- Candidate Activate Function

$$
\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h}),
$$

## Hidden State

![Untitled2](https://github.com/thiagorossener/jekflix-template/assets/47301926/e2c491bb-4e05-4c8b-b9b3-489999ca47eb)

Finally, we need to incorporate the effect of the update gate $Z_t$.

This determines the extent to which the new hidden state
$H_t \in \mathbb{R}^{n \times h}$ matches the old state $H_{t-1}$

compared with how much it resembles the new candidate state $\tilde{H}_t$.

The update gate $Z_t$ can be used for this purpose, simply by taking elementwise convex combinations of $H_{t-1}$ and $\tilde{H}_t$.

This leads to the final update equation for the GRU:

$$
\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.
$$

## 3.1 Customizing the GRU Model

Session-based 추천을 위해 GRU기반의 모델을 사용하였으며, 모델의 input은 session의 actual state을 받아서 session내 the item of the next event을 출력합니다.

session의 state라는 것은 `the item of the actual event` or `the events in the session` 로 볼 수 있으며, session내에서 어떤 정보를 사용하는지에 따라서 구분됩니다.

- `the item of the actual event`
  전자의 경우, 1-of-N 인코딩 기법이 사용되며, 입력 벡터의 길이는 아이템의 개수와 일치합니다. active상태에 따라 아이템은 1과 0으로 표현됩니다.
- `the events in the session`
  후자의 경우, 이러한 표현의 weighted sum을 사용하며, event의 발생 시점에 따라 조정됩니다. **즉 session 기반 추천시스템에서는 사용자의 최근 행동이 더 중요할 수 있으므로, 이전에 발생한 event는 현재 추천에 더 적은 영향을 미치게 됩니다.**
  input vector는 안정성과 메모리 관점에서 Normlize하는 것이 도움이 되었다고 합니다. 또한 임베딩 레이어를 추가하여 실험하였지만, 항상 1-of-N 인코딩 기법이 항상 더 나은 성능을 보였습니다.

네트워크의 핵심은 GRU layer이며, output사이에 Feedforward layer를 추가할 수 있습니다.

네트워크에서는 여러개의 GRU layer를 사용할 수 있으며, layer를 쌓을수록 성능이 향상되었습니다.

output은 session내 각 item들에 대해 preference(likelihood)를 예측합니다.

![gru](https://github.com/thiagorossener/jekflix-template/assets/47301926/58479499-7336-4702-9e4d-61354378f4e0)

### 3.1.1 SESSION-PARALLEL MINI-BATCHES

NLP 영역에서의 RNN은 문장 내 단어들에 대해 window size를 sliding하면서 min-batch로 구성하는 sequential mini-batch 기법을 주로 사용합니다.

하지만 이러한 기법은 추천 테스크에는 적합하지 않았는데 그 이유는 아래와 같습니다.

- **session의 길이가 일반적인 문장보다 훨씬 다양합니다.(2 ~ 수백 개 events)**
- **우리의 목표는 session이 시간에 따라 어떻게 변화하는지 파악하는 것이기 때문에 부분마다 자르는 것은 합리적이지 않습니다.**

따라서 session-parallel mini-batches라는 방법을 사용했습니다.

1. session의 순서를 나타내기 위해 각 session들을 정렬
2. 그 다음 mini-batch의 형태를 구성하기 위해서 X개의 session에서 first event만을 사용합니다.
   _(여기서 출력은 session의 second event입니다.)_

만약 한 session이 종료되면, 그 위치에 사용 가능한 다음 session이 배치됩니다.
session들은 독립적이라고 가정하기 때문에 switch되면, hidden state가 초기화 됩니다.

![mini](https://github.com/thiagorossener/jekflix-template/assets/47301926/b1b9524a-2205-44f2-8774-070e2f394864)

### 3.1.2 Sampling On the Output

몇 백만개의 item이 존재하는 경우를 생각해보면, 매번 각 단계에서 모든 item에 대한 score를 계산시, 복잡성이 number of items와 number of events의 곱으로 증가하게 됩니다. 이러한 계산은 실제 환경에서 사용하기에 비현실적일 수 있습니다.

따라서 output을 sampling하여 일부 item에 대해서만 score를 계산하였습니다. 즉 일부 item들에 대해서만 weight가 업데이트됩니다.

Positive Sample(desired output)뿐만 아니라 Negatvie sample에 대해서도 확률을 계산하고, Positive Sample이 높은 순위를 갖도록 weight를 수정해야 합니다.

missing event에 대한 자연스러운 해석은 사용자가 item의 존재를 몰랐기에 interaction이 없었다고 할 수 있습니다. 그러나 낮은 가능성으로 item을 알지만 선호하지 않기에 interaction 하지 않았던 것이라고도 할 수 있습니다.

item은 인기가 높을수록, 사용자가 알고 있을 가능성이 높기 때문에 missing event는 선호하지 않다는 것을 나타낼 가능성이 높습니다.

**따라서 item의 인기도에 비례하여 sampling해야 할 필요성이 있습니다.**

이를 위해 train 데이터에 대해 sampling을 따로 생성하는 대신에 mini-batch의 다른 아이템을 negative sample로 설정합니다. 즉 mini-batch 내의 다른 train 데이터에서 item을 선택하면, 그 item이 mini-batch에 포함될 확률이 인기도에 비례하므로, popularity-based sampling입니다.

위와 같은 방법은 sampling을 생략함으로써 계산 시간을 줄일 수 있으며, faster matrix operations을 가능하게 합니다.

### 3.1.3 RANKING LOSS

추천 시스템의 핵심은 **사용자와 관련성 있는 items에 대해 순위를 부여하는 것**입니다. 이는 단순 분류 작업으로 볼 수도 있지만, 일반적으로 "learning-to-rank" 접근법이 다른 방법보다 더 우수한 성능을 보입니다.

- **Pointwise Ranking**

  각 아이템의 score나 rank를 서로 독립적으로 추정하며, loss는 item의 rank가 높게 산출되도록 정의됩니다.

- **Pairswise Ranking**

  Positive item과 Negative item을 Pair로 구성하여 score나 rank를 비교합니다. loss는 Positive가 Negative item보다 rank가 높게 산출되도록 정의됩니다.

- **Listwise Ranking**

  모든 item들에 대해 score와 rank를 계산하고, perfect ordering에 기반하여 비교합니다. sorting이 수반되기 때문에 계산 측면에서 비용이 더 요구되며, 자주 사용되지 않습니다. 만약 관련된 item이 1개뿐일 경우, pairwise ranking으로 해결합니다.

실험적으로 Pointwise Ranking방법은 네트워크 상에서 불안정하였으며, Pairwise ranking loss기반 방식들이 상대적으로 성능면에서 우수하였으며, 아래 2가지 방법이 사용되었습니다.

- **BPR(Bayesian Personalized Ranking)**

  $$
  L_s=-\frac{1}{N_S} \cdot \sum_{j=1}^{N_S} \log \left(\sigma\left(\hat{r}_{s, i}-\hat{r}_{s, j}\right)\right)
  $$

  - ${N_S}$: sample size
  - $\hat{r}_{s, k}$: the score on item $k$ at the given point of the session, $i$ is the desired item (next item in the session) and $j$ are the negative samples.

- **TOP1**

  TOP1 Ranking loss는 테스크를 위해 설계되었으며, regularized approximation of the relative rank of the relevant item를 나타냅니다.
  우선 The relative rank of the relevant item란 무엇인지를 살펴봅니다.

  $$
  \frac{1}{N_S} \cdot \sum_{j=1}^{N_S} I\left\{\hat{r}_{s, j}>\hat{r}_{s, i}\right\} = \frac{1}{N_S} \cdot \sum_{j=1}^{N_S} \sigma\left(\hat{r}_{s, j}-\hat{r}_{s, i}\right)
  $$

  $I\{\cdot\}$는 sigmoid로 근사되었습니다.
  $\hat{r}_{s, j}$의 score가 증가하면, $\hat{r}_{s, j}-\hat{r}_{s, i}$는 증가하게 됩니다. sigmoid 함수의 특성상, 이 입력 값이 증가하면 출력 값도 증가하게 됩니다.

  따라서 TOP1 loss fucntion은 Positive item의 score가 높아지도록 parameter들을 조정하면서 최적화를 하게 됩니다. 하지만 item이 이미 positive example로 score가 높아진 경우, negative example에서도 score가 계속 높아지는 경향이 있기 때문에 불안정합니다.

  따라서 Negative item의 경우, 0에 가깝게 유지되도록 regularization term을 도입하면서 아래와 같이 업데이트하였습니다.

  $$
  L_s=\frac{1}{N_S} \cdot \sum_{j=1}^{N_S} \sigma\left(\hat{r}_{s, j}-\hat{r}_{s, i}\right)+\sigma\left(\hat{r}_{s, j}^2\right)
  $$

# 4 EXPERIMENTS

2개의 데이터셋에 대하여 인기있는 Baseline모델들과 제안된 GRU4Rec모델로 평가하였습니다.

1. RecSys Challenge 2015(YooChoose)

- challenge의 training set만 사용하며, click events만 고려
- Session의 length가 1인 경우는 Filtering
- 약 6개월의 데이터가 학습에 사용되며, 7,966,257 sessions of 31,637,239 clicks on 37,483 items를 포함
- 그 다음 날의 session부터 test set으로 사용하기 때문에 각 session은 Train / test에 할당되며, 중간에 분할되지 않음
- collaborative filtering 방법의 특성으로 인해, test set에서 클릭된 item이 train set에 없는 경우 해당 item은 제외됩니다.

2. a Youtube-like OTT video service platform(VIDEO)

- 일정 시간 이상으로 시청된 동영상에 대한 event를 수집
- 특정 지역만을 대상하였으며, 수집 기간은 2개월이 조금 되지 않음
- 다양한 알고리즘에 기반한 item-to-item 추천을 각 동영상 왼쪽 sidebar에 제공
- bot에 의해 생성되었을 가능성이 있는 매우 긴 session의 경우는 Filtering
- train set은 마지막날을 제외한 모든 데이터로 구성되며, ∼ 3 million sessions of ∼ 13 million watch events on 330 thousand videos
- test set은 마지막날의 session으로 구성되며, 37 thousand sessions with ∼ 180 thousand watch events

평가는 session의 event를 하나씩 제공하고, next event에서 item의 rank를 확인하는 방식으로 이루어집니다. item들은 score에 따라 내림차순으로 정렬하여 나타난 position이 rank입니다.

RSC15의 경우, train set의 모든 37,483 items들에 대해서는 rank가 부여되지만 VIDEO에서는 item이 너무 많기 때문에 rank를 부여하는 것이 비현실적입니다.

따라서 가장 인기있는 30,000 items 중에서 원하는 item들에 대해서만 rank를 산정하였습니다. 이는 popularity based pre-filtering으로 볼 수 있으며, 인기도가 낮은 item의 경우 평가에 미치는 영향이 크지 않기 때문입니다.

## 4.1 Baseline

비교를 위해 Baseline으로 사용된 모델은 아래와 같습니다.

- POP: Popularity predictor that always recommends the most popular items of the training
  set.
- S-POP: This baseline recommends the most popular items of the current session.
- Item-KNN: Items similar to the actual item are recommended
- BPR-MF: one of the commonly used matrix factorization methods. It optimizes for a pairwise ranking objective function (see Section 3) via SGD

![table1](https://github.com/thiagorossener/jekflix-template/assets/47301926/0aa0b40e-1d24-4ce2-91b2-c4f84216ed00)

## 4.2 PARAMETER & STRUCTURE OPTIMIZATION

- 각 dataset과 loss function에 대해서 random search를 기반으로 100개의 실험을 진행하여 parameter을 optimize
- 모든 경우에서 hidden unit의 수는 100으로 설정되었으며, hidden layer의 수는 조금씩 차이가 있습니다. paramter는 uniform 분포에 의해 initalize되며, 범위는 matrix의 크기에 의존합니다.
- Optimizer로는 rmsprop과 adagrad를 사용하여 실험되었으며, adagrad가 더 우수한 결과를 보여주었습니다.
- Pointwise ranking 기반 loss들은 regularizationd을 적용해도 불안정한 모습을 보였으며, Pairwise ranking 기반의 BPR과 TOP1은 안정적이며, 성능적으로도 가장 우수하였습니다.

![table2](https://github.com/thiagorossener/jekflix-template/assets/47301926/84fdf862-53d7-4d2d-b5de-194513de0dcd)

또한 single layer을 가지는 GRU가 성능에서 가장 뛰어났으며, 이는 정확하지 않지만 session이 일반적으로 짧은 lipespan을 가지기 때문에 multiple time scale이 필요하지 않다고 추측해볼 수 있습니다.

- item에 대해서 Embedding을 사용하는 것보다 1-of-N 인코딩 기법이 성능이 약간 우수하였습니다.
- 모든 previous event를 사용하는 것과 preceding one event를 사용하는 것에서 성능에 큰 차이는 없었습니다. 이는 GRU가 long and short term memory를 모두 가지고 있다는 걸 생각해보면 납득할 수 있습니다.
- gru layer이후, feed-forward layers를 추가하는 것도 크게 도움이 되지는 않았습니다.
- gru layer에 대해 hidden size를 키우는 것은 성능에 개선이 있었습니다.
- 또한, output layer에 대한 activate function으로 tanh를 사용하는 것이 좋았습니다.

## 4.3 Result

![table3](https://github.com/thiagorossener/jekflix-template/assets/47301926/9e9e5e4f-4115-49e5-90e0-a4796063aec5)
