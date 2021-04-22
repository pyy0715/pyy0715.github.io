---
date: 2019-12-14 18:39:28
layout: post
title: Wide & Deep Learning for Recommender Systems 리뷰
subtitle: Paper Review
# description: Paper Review
math: true
image: https://res.cloudinary.com/dm7h7e8xj/image/upload/v1559822138/theme9_v273a9.jpg
optimized_image: https://res.cloudinary.com/dm7h7e8xj/image/upload/c_scale,w_380/v1559822138/theme9_v273a9.jpg
category: Recommender System
tags:
    - Recommender System
    - RecSys
    - Wide and Deep
    - Paper Review
author: pyy0715
---

# Wide & Deep Learning for Recommender Systems

> [Paper](https://arxiv.org/pdf/1606.07792.pdf%29/)
>
> [Google Blog](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)

## ABSTRACT

입력 변수가 매우 sparse한 경우, 회귀/분류 문제를 풀기 위해 비선형 변수로 변환 후 일반화 선형 회귀모델을 사용합니다.
외적으로 변수 변환을 하여, **Wide**하게 변수간의 상호작용을 **Memorization** 한 경우 효과적이고 해석이 용이해집니다. 반면 일반화의 경우 더 많은 피쳐 엔지니어링의 노력이 필요합니다.
DNN **(Deep)**은 sparse한 변수에 대해 학습시킨 저차원 임베딩을 통해 간단한 피쳐 엔지니어링으로 전에 나오지 않았던 변수 조합에 대한 일반화를 보다 더 잘할 수 있다.
그러나 임베딩을 통한 DNN은 과대적합이 될 수 있으며, 유저와 아이템의 관계가 매우 sparse하고 차원이 높다면, 관계가 없는 아이템을 추천해줄 수도 있다.

이 논문은 Memorization과 일반화 두마리의 토끼를 잡기 위해 추천 시스템에 Wide & Deep 학습, 즉 wide 선형 모형과 deep 신경망을 함께 훈련시킨 방법론을 제시합니다.
10억 명이 넘는 활성 사용자와 100만 개가 넘는 앱을 보유한 상용 모바일 앱 스토어 Google Play에서시스템을 구축하고 평가했으며, 온라인 실험 결과에 따르면 Wide & Deep은 wide 선형 모형만 사용한 것과 deep 신경망만 사용한 것 대비해서 앱 가입을 크게 증가시켰다.


## 1. INTRODUCTION

추천 시스템은 사용자 및 맥락 정보 집합이 입력 쿼리이고 출력이 아이템 순위를 매긴다는 점에서 검색 순위 시스템 일종으로 볼 수 있습니다. 추천 작업은 쿼리가 주어졌을 때 데이터베이스에서 관련 품목을 찾고 클릭 또는 구매 같은 특정 목표에 기반하여 품목 순위를 메깁니다.

추천 검색 시스템의 과제는 일반적인 검색 순위 문제와 마찬가지로 **Memorization**과 **Generalization**를 모두 달성하는 것입니다. 

**Memorization**은 동시에 빈발하는 품목 또는 변수를 학습하고 과거 이력에서 이용 가능한 상관관계를 뽑아내는 작업으로 정의됩니다.
한편, **Generalization**는 상관관계의 이행성(transtivity)에 기반하고 거의 발생하지 않은 새로운 변수 조합을 탐구합니다. **Memorization**에 근거한 추천은 보통 사용자가 이미 행동을 취했던 품목과 직접적으로 관련되어 있습니다. **Memorization과** 비교할 때, **Generalization**는 추천 품목 다양성이 향상되어집니다. 이 글에서는 Google Play 스토어 앱 추천 문제를 다루지만 일반적인 추천 시스템에도 적용해볼 수 있습니다.

기업의 대형 온라인 추천 및 순위 시스템에서는 로지스틱 회귀 같은 일반화된 선형 모형이 간단하고 해석하기 쉽기 때문에 널리 사용합니다. 이런 모델들은 종종 one-hot 인코딩을 사용하여 sparse 변수에 대해 모형을 훈련시킵니다. 예를 들자면 이진변수 `user_installed_app = netflix`는 사용자가 Netflix를 설치한 경우, 값 1을 가집니다.

**Memorization은**  sparse한 변수들에 대해 외적변환을 사용함으로써,  `AND(user_installed_app = netflix, impression_app = pandora)` 와 같이 사용자가 Netflix를 설치했고 이후 Pandora를 설치했다면값은 1을 가집니다. 이는 변수 쌍의 동시 발생이 목표 변수 레이블과 어떻게 연관되는지 설명해줍니다.  `AND(user_installed_category = video, impression_category = music)` 같이 세분화가 덜 된 변수를 Generalization에 사용해서 일반화할 수 있지만  많은 피쳐 엔지니어링 작업이 요구됩니다. 그러나 외적변환을 통한 Memorization의 한계점은 훈련 데이터에 나타나지 않은 쿼리 - 아이템 변수 쌍을 일반화하진 못한다는 점입니다.

factorization machine 또는 신경망 같은 임베딩 기반 모형들은 피쳐 엔지니어링에 대한 부담을 줄이면서 쿼리 및 품목 변수마다 저차원의 밀집 임베딩 벡터를 학습시켜 이전에 보지 못한 쿼리 - 아이템 변수 쌍을 일반화할 수 있습니다. 그러나 특정 선호도를 가진 사용자나 틈새 아이템과 같이 희소하고 차원이 높은 경우에는 쿼리 - 아이템 행렬에 대해서는 저차원 표현으로 학습이 어렵습니다.

위의 경우 대부분의 [쿼리 - 아이템] 간에 상호작용이 없음에도 밀집 임베딩은 모든 [쿼리 - 아이템] 쌍에 대해 0이 아닌 값을 예측하기 때문에 따라서 과대적합이 되거나 별로 관계없는 추천을 할 수 있습니다. 한편 외적 변수 변환을 통한 선형 모형은 훨씬 적은 수의 매개 변수로 이러한 “예외 규칙”을 Memorization 할 수 있습니다.

본 논문은 그림 1과 같이 선형 모형 구성 요소와 신경망 구성 요소를 함께 학습한 모형 안에서 Memorization 및 Generalization 모두를 달성할 수 있는 Wide & Deep 학습 프레임워크를 제시합니다.

![Untitled](https://user-images.githubusercontent.com/47301926/70879083-72235700-2007-11ea-935a-268c8f2d3ba4.png)

본 논문의 주된 기여는 다음과 같습니다.

- 입력값이 희소한 일반 추천 시스템을 위해 임베딩을 통한 피드-포워드 신경망과 변수 변환을 통한 선형 모형을 함께 훈련시키는 Wide & Deep 학습 프레임워크
- 10억 명 이상의 활성 사용자와 100만 개 넘는 앱이 있는 모바일 앱 스토어 인 Google Play에서 Wide & Deep 추천 시스템 제품화 구현 및 평가
- TensorFlow 고수준 API를 통한 오픈소스 구현

아이디어는 단순하지만 Wide & Deep 프레임워크는 모형 훈련 및 서비스 속도 요건을 만족시키면서 모바일 앱 스토어 앱 가입률이 크게 향상했습니다


## 2. RECOMMENDER SYSTEM OVERVIEW

![Untitled 1](https://user-images.githubusercontent.com/47301926/70879084-72235700-2007-11ea-8d99-edfb0cd0cf33.png)

앱 추천 시스템에 대한 개요가 그림 2에 나와있습니다. 사용자가 앱 스토어를 방문하면 사용자 본인과 맥락에 관련된 다양한 변수가 포함되어 쿼리가 생성됩니다. 추천 시스템은 사용자가 클릭이나 구매 같은 특정 동작을 수행할 수 있는 앱 목록(노출이라고도 함)을 반환합니다. 사용자 동작은 쿼리 및 노출과 함께 학습기를 위한 훈련 데이터로 기록됩니다.

데이터베이스에는 100만 개가 넘는 앱이 있기에 요구되는 서비스 대기 시간(대부분 O(10) 밀리세컨드) 이내로 모든 쿼리문마다 전체 앱에 점수를 산출하는 건 어렵습니다. 따라서 쿼리 수신 후 첫 번째 단계는 ***Retrieval*** 입니다. Retrieval(검색) 시스템은 일반적으로 기계 학습 모형과 사람이 정의한 규칙 조합과 같은 다양한 signal을 사용하여 쿼리문과 가장 일치하는 아이템의 짧은 목록을 반환합니다. 아이템의 후보 범위를 줄인 후 순위 시스템은 후보 아이템들의 점수 순위를 산출합니다.

점수는 대개 $P(Y \mid X)$, 변수 $x$가 주어졌을 떄 유저의 행동, $y$의 조건부 확률을 사용합니다.
즉 사용자 변수(예: 국가, 언어, 인구통계학적), 맥락 변수(예: 기기, 시간대, 요일)와 노출 변수(예: 앱 출시 후 경과 기간, 앱 통계 이력)를 포함하여 변수 $x$가 주어졌을 때 사용자 동작 각 레이블 $y$의 확률입니다. 본 논문은 Wide & Deep 학습 프레임워크를 사용한 순위 모형에 초점을 맞출 것입니다.

## 3. WIDE & DEEP LEARNING

### 3.1 The Wide Component

Wide 구성 요소는 일반화 선형 모델의 형태인 $y=w^x+b$의 형태를 갖습니다. 변수는 입력값과 변환된 변수를 포함합니다. 가장 중요한 변환 중 하나는 다음과 같이 정의되는 외적변환입니다.

![Untitled 2](https://user-images.githubusercontent.com/47301926/70879085-72235700-2007-11ea-9043-7e6f0fba96e9.png)

여기서 $c_{ki}$는 $i$번째 변수가 $k$번째 변환 ϕk의 일부이면 1이고 그렇지 않으면 0인 이진 변수입니다.

`AND(gender=female, language=en)`와 같이 gender=female, language=en인 경우는 1이고 그렇지 않으면 모두 0입니다. 이는 이진 변수 사이의 관계의 특징을 잡아주고 일반화 선형 모델에 비선형성을 더해줍니다.

### 3.2 The Deep Component

deep 구성 요소는 피드포워드 신경망으로 그림 1의 오른쪽 부분으로 확인할 수 있습니다. 범주형 변수에 대해서 원래 입력값은 문자열 변수(예: `language = en`)입니다. 이런 희소하고 고차원인 범주형 변수 각각은 임베딩 벡터라고 하는 저 차원의 밀집한 실수 값 벡터로  변환됩니다. 임베딩 차원은 일반적으로 O(10)에서 O(100) 수준으로 정합니다. 임베딩 벡터는 임의로 초기화된 후 모형 훈련 과정을 통해 최종 손실 함수를 최소화하도록 값이 훈련됩니다.. 이러한 저차원의 밀집한 임베딩 벡터는 포워드 과정 중 hidden layer로 fed 되어집니다. 구체적으로 각 hidden layer는 밑의 계산을 수행합니다.

![Untitled 3](https://user-images.githubusercontent.com/47301926/70879086-72bbed80-2007-11ea-8a98-0ba57f0fcf5a.png)

$l$은 layrer의 숫자이며, $f$는 활성화 함수로 ReLU를 사용하였습니다.

### 3.3 Joint Training of Wide & Deep Model

![Untitled 4](https://user-images.githubusercontent.com/47301926/70879087-72bbed80-2007-11ea-9911-084414807df6.png)

wide 구성 요소와 deep 구성 요소는 [로그 오즈](https://m.blog.naver.com/PostView.nhn?blogId=sw4r&logNo=221150181217&proxyReferer=https%3A%2F%2Fwww.google.com%2F) 가중치 합계를 예측치로 사용하기 위해 결합 되어집니다. 그리고 예측치는 joint 학습을 위해 로지스틱 손실 함수로 사용됩니다.

joint 훈련과 앙상블이 구별된다는 것을 유의하시길 바랍니다. 앙상블에서는 각각의 모델들은 독립적으로 훈련되어지며, 모델들의 예측치들은 결과값을 내기 위해 결합되어집니다.
대조적으로 joint훈련은  wide와 deep부분을 모두 고려하여 동시에 파라미터를 최적화 뿐만 아니라 훈련시 가중치 합계를 사용합니다.

앙상블의 경우 훈련이 분리되어 있으므로 합리적인 정확도를 얻기 위해 모형 각각이 좀 더 커야 한다(예: 더 많은 변수와 변수 변환). 이와는 다르게 공동 훈련의 경우 넓은 쪽은 전체 크기의 Wide 모형보다 적은 수의 외적변수 변환으로 Deep부분의 약점을 보완하기만 하면 됩니다.
Wide & Deep 모형 공동 훈련은 미니 배치 단위의 확률적 경사 하강법을 이용하여 출력 값 기울기를 모형 wide 쪽과 deep 쪽 동시에 역전파시킨다. 실험에서 모델의 wide 부분에 대한 최적화로 L1 정규화를 따르는 Follow-the-regularized-leader(FTRL) 알고리즘을 사용했고 deep 부분에 대해서는 AdaGrad를 사용하였습니다.

![Untitled 5](https://user-images.githubusercontent.com/47301926/70879081-718ac080-2007-11ea-8c18-16bdec6fd69a.png)

여기서 $Y$는 이진 값 클래스 레이블이고 $σ(⋅)$는 시그모이드 함수, $ϕ(x)$는 원래 변수 $x$의 외적 변수 변환, $b$는 편향값 입니다. wide는 Wide 모형의 모든 가중치 벡터이고 deep은 최종 출력 값 $a^{l_f}$에 적용한 가중치이다.

## 4. SYSTEM IMPLEMENTATION

앱 추천 파이프라인 구현은 3.3에 나와있는 그림3과 같이 데이터 생성, 모형 훈련 및 모형 서비스 같은 3단계로 구성됩니다.

### 4.1 Data Generation

이 단계에서는 일정기간 동안 사용자와 앱 노출 데이터를 사용하여 훈련 데이터를 생성합니다. 각 샘플은 노출 한 번에 해당한다. 레이블은 **앱 가입** 입니다. 즉, 노출한 앱을 설치하면 1이고 그렇지 않으면 0입니다.

범주형 문자열 변수를 정수 ID로 맵핑한 사전도 이 단계에서 생성합니다. 시스템은 설정한 최소 횟수 이상으로 발생하는 모든 문자열 변수에 대해 ID 공간을 계산합니다. 연속적인 실수 값 변수는 변수값 $x$를 누적 분포 함수 P(X≤x)에 연결하여 n개의 분위수로 나누어 [0,1]로 정규화 합니다. 정규화한 값은 $i$번째 분위수 값에 대해 $i−1$ / $n−1$입니다. 데이터 생성 동안 분위값의 경계를 계산합니다.

### 4.2 Model Training

![Untitled 6](https://user-images.githubusercontent.com/47301926/70879082-72235700-2007-11ea-9c1f-7a55ffcac608.png)

사용한 모델의 구조는 위의 그림과 같습니다.
훈련 중, Input layer는 훈련 데이터와 사전을 받아서 레이블과 함께 Sparse 또는 Dense한 변수를 생성합니다.  

wide 부분의 구성 요소는 사용자가 설치한 앱과 노출된 앱의 외적 변수 변환으로 구성합니다.
deep 부분의 구성 요소는  32차원 임베딩 벡터로 각 범주형 변수에 대해 학습합니다. 모든 임베딩을 밀집 변수와 연결하여 약 1,200차원 밀집 벡터를 생성한다. 연결한 벡터를 3개의 ReLU 층으로 전달하고 마지막으로 로지스틱 출력 단위로 전달합니다.

Wide & Deep 모형을 5천억 개가 넘는 샘플로 훈련시키고, 일련의 새로운 훈련 데이터가 수집될 때마다 모델을 다시 훈련시켜야 합니다. 그러나 매번 처음부터 재훈련시키는 건 계산 비용이 많이 들고 데이터 수집부터 서비스까지 시간이 많이 소요됩니다. 이 문제를 해결하기 위해 임베딩과 이전 모델의 가중치를 사용하여 새 모델 초기값을 설정하는 식의 warm-starting system를 구현하였습니다.
서버에 모델을 적재하기 전에 실제 트래픽을 처리하는데 문제가 없는지 테스트를 하였으며, 이전 모델에 비해 성능이 좋았졌는지 경험적으로 측정하였습니다.

### 4.3 Model Serving

모델을 훈련하고 검증이 끝나면 모델 서버에 모델을 적재합니다. 각 요청마다 서버는 앱 Retrival(검색) 시스템에서 앱 후보군을 수신하고 사용자 변수를 사용하여 후보 앱에 점수를 매깁니다. 앱은 가장 높은 점수부터 가장 낮은 점수까지 순위를 매기며 그 순서로 사용자에게 노출합니다. 점수는 Wide & Deep 모델에 대한 순방향으로 실행되면서 계산되어집니다.
10ms 단위로 각 요청을 처리하기 위해 추론 단계를 단일 배치로 후보 앱 전체에 점수를 매기는 대신 멀티스레딩 병렬 처리를 통해 미니 배치를 병렬로 돌려서 성능을 최적화했습니다.

## 5. EXPERIMENT RESULTS
생략

## 6. RELATED WORK

생략

## 7. CONCLUSION

추천 시스템에서 **Memorization**과 **Generalization**은 중요합니다. Wide linear 모델은 외적 변수 변환을 통해 희소 변수 간 상호 작용을 효과적으로 **Memorization**할 수 있지만 DNN은 저차원 임베딩을 통해 이전에 보이지 않던 변수 간 상호 작용을 일반화할 수 있다. 두 가지 모델 유형의 장점을 결합하기 위해 Wide & Deep 학습 프레임워크를 제시했다. 대규모 앱 스토어 Google Play 추천 시스템에서 프레임워크를 구축하고 평가했다. 온라인 실험 결과에 따르면 Wide & Deep 모델은 wide만 사용한 모형과 deep만 사용한 모델 대비해서 앱 가입률이 크게 향상하였다.

## 참고자료

[Wide & Deep Learning for Recommender Systems](http://hugrypiggykim.com/2018/03/29/wide-deep-learning-for-recommender-systems/)


[텐서플로우 튜토리얼](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/wide_and_deep/)

[Park-Ju-hyeong/Wide-Deep-Learning](https://github.com/Park-Ju-hyeong/Wide-Deep-Learning/blob/master/Wide%2526Deep%2BRecommendation-Final-Final.ipynb)
