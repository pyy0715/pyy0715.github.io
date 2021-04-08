---
date: 2021-03-21 18:39:28
layout: post
title: Session-Based Recommendations with Recurrent Neural Networks 리뷰
subtitle: Paper Review
# description: Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
math: true
image: https://images.unsplash.com/photo-1615224572819-61e7e440a7e5?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=624&q=80
optimized_image: <img_src="https://images.unsplash.com/photo-1615224572819-61e7e440a7e5?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=624&q=80" width="380">
category: Recommender System
tags:
    - Session-Based
    - GRU4Rec
author: pyy0715
---
# Session-Based Recommendations with Recurrent Neural Networks

> [Paper](https://arxiv.org/pdf/1511.06939.pdf)

# ABSTRACT

- 실제 추천시스템은 넷플릭스 처럼 사용자의 프로파일 대신에 짧은 Session 기반의 데이터로 추천을 해야 하는 문제에 직면하는데 이런 경우는 Matrix Factorization와 같은 접근법은 정확하지 않음.

- 이에 대한 해결법으로 RNN 기반의 session-based 추천 모델을 제안

# Introduction
지금까지 추천시스템에서 사용되었던 기본적인 방법들은 아래와 같다.

- `Factor Models`

    Matrix Completion문제로 유저와 아이템의 latent vector를 구해 내적값으로 missing value를 채움

    **Session단위에서는 사용자의 Profile을 이용하지 않기 떄문에 접근이 어려움**

- `Neighborhood Methods`

    Session 내 유저 또는 아이템간의 유사성을 계산하는 것을 목적으로 한다.

    따라서 Session단위 접근법에서는 광범위하게 사용될 수 있다.

E-commerce의 추천시스템은 왜 Session단위로 이루어져야 하는지에 대해서 설명
- 한 사용자의 모든 기록을 추적할 수는 없다(사생활 보호, cookie 데이터의 신뢰성 등)
- 같은 사용자의 연속적인 Session은 독립적으로 다루어야 한다.

결과적으로 E-commerce의 추천시스템은 상대적으로 간단하게 아이템 간의 유사성을 기반으로 하게 되었다. 이러한 방법은 효과적이지만 결국 사용자의 과거 클릭정보를 무시하게 된다.

따라서 저자는 RNN을 이용하여 Session 단위의 정교한 추천 시스템을 만드는 것이 목적이다. 위의 `Neighborhood Methods`에 기반하여 유저가 처음 클릭한 아이템으로 그 다음 아이템을 예측할 수 있는 모델, 즉 사용자가 관심을 가질 수 있는 상위 항목을 만들려고 한다.

논문에서 주의깊게 바라보아야 할 점은 아래의 3가지이다.

- **Sequential한 데이터에서 sparsity를 어떻게 해결하였는지?**
- **추천 Task를 위해 새로운 Ranking Loss는 무엇을 사용하였는지?**
- **click-stream 데이터 또한 상당히 크기 때문에 훈련 시간과 확장성을 어떻게 고려하였는지?**

# Related Work

## Session-Based Recommendation

기존 방법들의 경우, 사용자의 마지막 클릭 아이템을 기준으로 유사한 아이템을 찾기 때문에 과거 클릭 정보를 무시하게 되는 문제점이 있다.

이러한 문제점을 해결하기 위해 노력한 2가지의 방법을 소개한다.

- **MDP(Markov Decision Process)**

    MDP방법을 이용하여 아이템 간 전이확률을 기반으로 간단하게 다음 추천, Action이 무엇인지 계산할 수 있다.

    하지만 빠르게 아이템의 수가 많아지기 때문에, 즉 유저가 선택할 수 있는 폭이 넓어지기 때문에 MDP기반의 접근방법만으로는 state space를 점점 관리할 수 없게 된다. 

- **GFF(General Factorization Framework)**

    아이템에 대해서 두 종류의 latent vector를 사용한다. 하나는 `아이템 그 자체`와 다른 하나는 `session으로서의 아이템`을 사용한다.

    따라서 어떤 session은 `session으로서의 아이템`들의 평균으로 표현될 수 있다.하지만 session간에 순서를 고려하지 않는다.

## Deep Learning In Recommendation

Restricted Boltzmann Machines (RBM)은 Collaborative Filtering 모델들에서 유저와 이템의 Interaction을 바탕으로 우수한 성능을 나타내었습니다.

최근 Deep Model들은 음악이나 이미지같이 구조화되지 않은 컨텐츠에서 feature를 추출하기 위해 사용되어졌으며, 전통적인 CF들과 함께 사용되어져왔습니다.


# Recommendation with RNNs

RNN은 기본적으로 variable-length sequence한 데이터를 처리하도록 설계되었습니다.
RNN과 전통적인 피드-포워드 신경망 모델들의 차이는 신경망을 구성하는 unit내의 hidden state의 존재 유무입니다.

GRU는 vanishing gradient problem을 다루기 위해 설계되어진 정교한 RNN 모델입니다.
![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile7.uf.tistory.com%2Fimage%2F99F0EC3E5BD5F6460255CF)

## Customizing the GRU Model

Session 단위의 추천을 위해 GRU기반의 모델을 사용하였으며, 모델에 대한 입출력은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/47301926/111938366-1fd7bc80-8b0d-11eb-8cec-3546acc26b74.png)


> Inputs: actual state of the session
>
> Outputs: the item of the next event in the session

session의 state는 `the item of the actual event or the events in the session`로 두 가지로 표현될 수 있습니다. 이 부분이 좀 이해하기 어려운데 예를 들어서 설명해보겠습니다.

1. **the item of the actual event**
전자의 경우, 1-of-N 인코딩 기법이 사용되며, 입력 벡터의 길이는 아이템의 개수와 일치하며, active 상태에 따라 아이템은 1과 0으로 표현됩니다. 쉽게 생각하면 사용자가 클릭했던 아이템들 중 실제 구매로 이어진 아이템을 나타내기 위해 표현되었습니다.

2. **the events in the session**
후자의 경우, weighted sum으로 표현되어지는데 event의 발생 시점에 따라 조정됩니다.
사용자가 클릭했던 아이템을 고려하면서도 이에 대한 시간 순서, 즉 최근 클릭한 아이템이 가장 큰 weight를 가질 수 있도록 표현하기 위함입니다.
이런 방법은 기존 RNN이 가지는 장기의존성 문제를 해결할 수 있다고 합니다. 안정성을 위해 입력 벡터는 정규화를 수행합니다. 

또한 아이템에 대한 임베딩 레이어를 추가하면서 실험하였는데 항상 1-of-N 인코딩 기법이 성능이 우수하였다고 말합니다.

GRU네트워크의 핵심은 마지막 gru 레이어와 output 레이어 사이에 추가한 feedforward layer 입니다. GRU 레이어들을 여러개 쌓을 수록 성능이 향상되었다고 실험 결과에서 말하고 있습니다.

## 3.1.1 SESSION-PARALLEL MINI-BATCHES
NLP영역에서의 RNN은 문장 내 단어들의 window_size를 이동하면서  sequential mini-batch 기법을 주로 사용합니다. 즉, 설정 사이즈에 맞춰서 부분적으로 예측합니다. 

하지만 이러한 기법은 추천 Task에는 적합하지 않았는데 그 이유는 아래와 같습니다.
- **session의 길이가 일반적인 문장보다 훨씬 다양하다.**
- **목표는 session이 어떻게 변화하는지 포착하는 것이다. 그래서 부분마다 자르는 것은 합리적이지 않다고 한다.**

그에 대한 대안으로 session-parallel mini-batches를 사용했다고 합니다. 방법은 다음과 같습니다.

![image](https://user-images.githubusercontent.com/47301926/113396407-76c77680-93d6-11eb-9d67-861e8afc151f.png)

1. 각 Session들을 정렬시킨다.
2. 그 다음 mini-batch의 형태를 구성하기 위해서 $X$개의 session에서 첫 event만을 사용합니다.
3. 그 다음 event들로 mini-batch를 구성합니다.

만약 한 session이 종료되면 사용 가능한 다음 session이 배치됩니다. Session들은 독립적이라고 가정하기 때문에 병렬적으로 구성할 수 있습니다.

## 3.1.2 Sampling On the Output

아이템의 수가 너무 많다는 것을 고려하면, 매번 각 스텝에서 모든 아이템에 대한 선호 확률을 계산하는 것은 사용할 수 없습니다. 따라서 output으로 나올 아이템을 sampling하고, 아이템에 대한 small subset을 구성함으로써 선호 확률을 구하였습니다.

이는 높은 순위를 가지도록 원하는 일부 아이템과 함께 Negatvie sample에 대해서도 확률을 계산하여, 우리가 원하는 아이템이 높은 순위를 갖도록 가중치를 업데이트해야 합니다.

missing event에 대한 자연스러운 해석은 사용자가 아이템의 존재를 몰랐기에 상호작용이 없었다고 할 수 있습니다. 그러나 낮은 가능성으로 아이템을 알지만 선호하지 않기에 상호작용 하지 않았던 것이라고도 할 수 있습니다.

어떠한 아이템은 인기가 높을수록, 사용자가 알고 있을 가능성이 높기 때문에 missing event로 선호하지 않을 가능성이 높습니다. 따라서 선호도에 비례하여 아이템을 sampling을 할 필요성이 있습니다.

각 train 데이터에 대해서 sample을 따로 생성하는 대신에, negative sample로의 역할을 할 수 있도록 mini-batch 내의 다른 train 데이터에 대한 아이템으로 설정합니다. 이는 mini-batch의 다른 train 데이터의 아이템 또한 선호도에 비례하기 때문입니다. 또한 따로 sampling을 구하지 않기 떄문에 계산 시간을 효율적으로 줄일 수 있습니다.



## 3.1.3 RANKING LOSS
업데이트 예정

# 4 EXPERIMENTS
업데이트 예정

# 5 CONCLUSION & FUTURE WORK
업데이트 예정

