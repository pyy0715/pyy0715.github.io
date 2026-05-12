---
date: 2026-05-12 00:00:00
layout: post
title: "Sparse Computation and Memory Management for Efficient LLM Inference"
type: concept
math: true

category: AI & ML
tags:
  - LLM
  - MoE
  - PagedAttention
  - vLLM
  - Inference
author: pyy0715
---

## Introduction

[이전 글](/posts/optimized-atteiton-on-inference/)에서는 MQA, GQA, FlashAttention을 통해 Attention 메커니즘의 메모리와 연산 병목을 어떻게 해결하는지 살펴봤습니다. Attention 쪽 비용이 줄어들고 나면, LLM 추론에서 비중이 커지는 다른 두 병목이 보입니다.

하나는 **FFN(Feed-Forward Network) 레이어의 파라미터와 연산량**입니다. Transformer 한 블록에서 Attention의 $Q, K, V, O$ 선형 변환은 $4d^2$의 파라미터를 차지하는 반면, FFN은 $W_1 \in \mathbb{R}^{d \times 4d}$와 $W_2 \in \mathbb{R}^{4d \times d}$로 약 $8d^2$를 차지합니다. 즉 블록 파라미터의 약 2/3가 FFN이며, LLaMA 3 70B처럼 dense하게 설계된 대형 모델에서는 전체 파라미터에서 FFN이 차지하는 비중이 80% 이상으로 올라갑니다.

다른 하나는 **KV Cache 메모리의 공간 낭비**입니다. 이전 글에서 KV Cache가 시퀀스 길이에 선형 비례하여 커진다는 점을 다뤘는데, 실제 프로덕션 환경에서는 그 안에서 추가적인 낭비가 발생합니다. SOSP 2023에서 발표된 [vLLM 논문](https://arxiv.org/abs/2309.06180)(Kwon et al.)은 기존 LLM 서빙 시스템에서 실제 토큰을 저장하는 데 사용되는 KV Cache 메모리가 전체 할당량의 20.4 ~ 38.2% 수준이라고 측정했습니다. 나머지 60% 이상은 메모리 단편화로 인해 사용되지 못한 채 남아 있다는 의미입니다.

이 두 병목을 각각 다루는 기법이 **Mixture of Experts(MoE)**와 **PagedAttention(vLLM)**입니다. 본 글에서는 두 기법이 제안된 논문을 차례로 살펴보며, 수식과 아키텍처, 그리고 실제 모델 수치까지 정리해보겠습니다.

> 이 글에서 다루는 기법은 이전 글의 MQA, GQA, FlashAttention과 서로 다른 축을 공략합니다. MoE는 모델 아키텍처 차원에서, PagedAttention은 서빙 시스템 차원에서 작동하며, Attention 메커니즘 자체의 최적화와는 성격이 다릅니다. 현대 프로덕션 스택에서는 이 기법들이 함께 조합되어 사용됩니다.

## Background: Why FFN Becomes the Next Bottleneck

MoE 이야기를 본격적으로 시작하기 전에, 왜 FFN이 다음 최적화 대상이 되는지 짚어보겠습니다. Hidden dimension을 $d$, 시퀀스 길이를 $N$이라 할 때, Transformer 한 블록의 연산 비용은 다음과 같이 정리됩니다 (행렬곱 leading-order 기준, bias와 LayerNorm 같은 부수항은 생략).

| 구성 요소 | 곱셈 대상 | 파라미터 수 | 토큰당 FLOPs |
|---|---|---|---|
| Attention 선형 변환 | 입력 × 가중치 ($W_Q, W_K, W_V, W_O$) | $4d^2$ | $8d^2$ |
| Attention Score/Output | 텐서 × 텐서 ($QK^\top$, $\cdot V$) | 0 | $4Nd$ |
| FFN | 입력 × 가중치 ($W_1 \in \mathbb{R}^{d \times 4d}$, $W_2 \in \mathbb{R}^{4d \times d}$) | $8d^2$ | $16d^2$ |

표의 첫 행과 셋째 행은 입력을 **학습된 가중치 행렬**과 곱하는 연산이고, 둘째 행은 입력에서 만들어진 **텐서끼리** 곱하는 Attention 메커니즘의 핵심 연산입니다. 둘째 행만 학습 파라미터가 없고 시퀀스 길이 $N$에 비례합니다.

FFN은 블록 전체 파라미터의 약 2/3를 차지하고, 토큰당 FLOPs도 Attention 선형 변환의 2배입니다. Score/Output 항은 $N$에 비례하지만 디코드 단계에서는 $N$이 작아, 결국 FFN이 토큰당 연산의 대부분을 차지합니다. 따라서 GQA로 KV 헤드를 줄이고 FlashAttention으로 HBM 왕복을 줄여도, FFN을 손대지 않으면 더 가속할 여지가 줄어듭니다.

MoE의 핵심 아이디어는 단순합니다. **FFN을 $N$개의 전문가(Expert)로 늘리되, 각 토큰마다 그중 $k$개($k \ll N$)만 활성화**하는 것입니다. 그러면 총 파라미터 수는 늘어나 모델 용량(capacity)이 커지지만, 토큰당 실제 연산량은 작은 모델 수준으로 유지됩니다.

![MoE Layer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/00_switch_transformer.png)
*Switch Transformer의 MoE 레이어 구조. 라우터가 각 토큰을 하나의 전문가로 보냅니다. (https://huggingface.co/blog/moe)*

## Sparsely-Gated MoE

### Original Formulation

ICLR 2017의 [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538) 논문에서 Shazeer 등은 LSTM 사이에 최대 137억 파라미터의 MoE 레이어를 끼워 넣었습니다. 그 결과 모델 용량은 1000배 이상 늘었지만, 토큰당 연산량은 비슷한 수준으로 유지되었습니다.

작동 방식은 간단합니다. MoE 레이어 안에는 전문가 $E_1, E_2, \ldots, E_N$이 있고, 그 위에 어떤 전문가를 쓸지 결정하는 작은 신경망인 **라우터(Router) 또는 게이팅 함수(Gating function) $G$**가 있습니다. 입력 토큰 $x$가 들어오면 라우터가 각 전문가에 대해 가중치 $G(x)_i$를 계산하고, 출력은 그 가중합으로 만들어집니다.

$$y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$

여기서 핵심은 $G(x)_i = 0$인 전문가는 아예 계산하지 않는다는 점입니다. 즉 $N$개 중 가중치가 0이 아닌 일부 전문가만 forward pass에 참여하므로, 전문가 수가 늘어나도 토큰당 실제 연산량은 작게 유지됩니다.

그렇다면 라우터는 어떻게 일부 전문가만 골라낼까요? Shazeer 등은 **Noisy Top-K Gating**이라는 방법을 제안했습니다. 이름 그대로 노이즈를 더한 뒤 상위 $k$개만 남기는 방식이며, 세 단계로 나누어 보면 직관이 잡힙니다.

**1단계. 각 전문가에 대한 점수 계산**

$$s_{i} = (x W_g)_{i}$$

그냥 일반적인 Linear 레이어입니다. W_g는 d × N 크기의 학습 가능한 가중치이고, 입력 x를 N차원으로 변환하면 각 차원이 전문가 i의 점수 s_i가 됩니다. 여기까지는 평범한 분류기와 똑같습니다. x가 어느 전문가에 가야 할지를 점수로 매기는 단계입니다.

**2단계. 점수에 노이즈 추가**

$$H(x)_{i} = (x W_g)_{i} + \mathcal{N}(0, 1) \cdot \mathrm{Softplus}((x W_{\mathrm{noise}})_{i})$$

세 부분으로 분해됩니다.

- 첫 번째 항: 1단계에서 구한 원래 점수
- 곱셈의 앞쪽 표준 정규분포 항: 평균 0, 분산 1의 가우시안 분포에서 뽑은 랜덤값
- 곱셈의 뒤쪽 Softplus 항: 노이즈의 세기. 별도의 학습 가능한 가중치 W_noise로 정해지며, Softplus를 통과해 항상 양수가 됨

즉 노이즈의 세기조차 학습 대상이라는 점이 핵심입니다. 모델은 어떤 입력에 대해 라우팅 결정을 더 무작위에 가깝게 둘지, 아니면 거의 결정적으로 둘지를 스스로 배워갑니다. 노이즈를 더하는 이유는 두 가지입니다. 학습 초기에 라우터가 특정 전문가에 고착되지 않도록 탐색을 유도하고, 모든 토큰이 한 전문가로 쏠리지 않도록 부하 균형에도 도움을 줍니다.

**3단계. 상위 K개만 남기고 softmax**

$$G(x) = \mathrm{Softmax}(\mathrm{KeepTopK}(H(x), k))$$

KeepTopK는 단순한 마스킹 함수입니다. 점수 H(x) 중 상위 k개만 그대로 두고 나머지는 음의 무한대로 만듭니다. 그 다음 softmax를 취하면 음의 무한대였던 값들은 자동으로 0이 되어, 정확히 k개만 0이 아닌 값으로 살아남습니다. 이렇게 만들어진 G(x)가 1단계 가중합의 가중치가 되고, 0이 아닌 k개의 전문가만 실제로 계산됩니다.

마지막으로 한 가지 디테일이 있습니다. 저자들은 $k$를 1이 아니라 최소 2 이상으로 두어야 한다고 설명합니다. $k=1$이면 라우터는 선택한 전문가의 출력 한 가지만 보게 되어, 다른 전문가를 골랐다면 결과가 더 좋았을지를 비교할 수 없습니다. 즉 라우터의 가중치를 어떻게 조정해야 할지에 대한 학습 신호가 끊깁니다. 두 전문가를 동시에 활용해야 라우터에 의미 있는 그래디언트가 흐른다는 것이 논문의 설명입니다 (이 제약을 뒤에서 살펴볼 Switch Transformer가 다른 방식으로 뒤집습니다).

## GShard and Switch Transformer

### GShard: Top-2 Routing

Google의 [GShard 논문](https://arxiv.org/abs/2006.16668)은 Sparsely-Gated MoE를 Transformer에 이식해 600B 파라미터 다국어 번역 모델을 2048 TPU v3로 4일 만에 학습시켰습니다. 이때 도입된 두 가지 변경이 이후 MoE 설계의 표준이 됩니다.

![GShard MoE Transformer Encoder](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/02_moe_block.png)
*GShard의 MoE Transformer 인코더 구조. 블록 두 개당 한 번씩 FFN이 MoE 레이어로 치환되고, 토큰이 Top-2 라우팅을 통해 두 전문가로 디스패치됩니다. (https://huggingface.co/blog/moe)*

첫째, 트랜스포머 블록을 한 칸씩 건너뛰면서 FFN을 MoE 레이어로 치환했습니다. 즉 모든 블록의 FFN을 바꾸는 것이 아니라, 짝수 번째(또는 홀수 번째) 블록에서만 FFN을 MoE로 교체하고 나머지 블록은 일반 FFN을 그대로 사용합니다.

둘째, **Expert Capacity** 개념을 도입했습니다. 한 전문가가 한 배치에서 받을 수 있는 토큰의 최대 개수를 미리 정해두는 것입니다.

$$C = \left\lceil \frac{T}{N} \right\rceil \cdot \text{capacity_factor}$$

여기서 $T$는 배치 내 전체 토큰 수, $N$은 전문가 수입니다. 토큰이 전문가에게 균등하게 분배되면 각 전문가가 받는 토큰 수는 $T / N$이 됩니다. 여기에 capacity_factor(보통 1.25 ~ 2.0)를 곱해 약간의 여유를 둔 값이 $C$입니다. 

어떤 전문가가 이미 $C$개의 토큰을 받았는데 또 토큰이 라우팅되면 그 토큰은 드롭(overflow)되어 잔차 연결(residual connection)로 우회합니다. 텐서 컴파일러가 정적인 텐서 모양을 요구하기 때문에, 즉 각 전문가의 입력 텐서 크기를 컴파일 타임에 고정해야 하기 때문에 도입된 메커니즘입니다.

이 용량 개념 위에서 Top-2 라우팅이 작동합니다. 1순위 전문가 $e_1$은 점수가 가장 높은 전문가로 결정론적으로 선택합니다. 2순위 전문가 $e_2$는 점수가 두 번째로 높은 전문가로 정한 뒤, **그 전문가를 실제로 디스패치할지 말지를 정규화된 게이트 가중치 $g_2$에 비례한 확률로 결정**합니다. 즉 $g_2$가 작으면 2번째 전문가는 호출하지 않고 그 토큰은 Top-1로 처리됩니다. 의사코드로 보면 다음과 같습니다.

```
g1, g2 ← top_2(softmax(score))
g1, g2 ← g1/(g1+g2), g2/(g1+g2)        # 정규화: g1 + g2 = 1
r ← uniform(0, 1)

# e1로 디스패치
if count[e1] < C:
    e1에 토큰 디스패치, count[e1] += 1

# e2로 디스패치 (조건부)
if count[e2] < C and 2·g2 > r:
    e2에도 디스패치, count[e2] += 1
```

`2·g2 > r` 조건의 2는 importance sampling 보정입니다. $g_2$는 정규화 후 보통 0.5 이하의 값이므로 단순히 `g_2 > r`만 쓰면 2번째 전문가가 거의 호출되지 않습니다. Top-2 환경에서 토큰당 평균적으로 두 전문가가 호출되도록 만들기 위해 곱하기 2가 들어갑니다.

논문이 이 설계를 택한 이유는 두 가지입니다. 첫째, 가중치가 작은 2순위 전문가는 어차피 기여가 미미하므로 호출을 생략해 통신과 연산 비용을 아낄 수 있습니다. 둘째, 확률적 디스패치가 특정 전문가로의 쏠림을 줄여 부하 균형에 도움을 줍니다.

### Switch Transformer: Top-1으로 단순화

Fedus, Zoph, Shazeer가 발표한 [Switch Transformer 논문](https://arxiv.org/abs/2101.03961)은 라우팅을 한 단계 더 단순화합니다. 논문은 다음과 같이 정리합니다.

> *"We route to only a single expert. We show this simplification preserves model quality, reduces routing computation and performs better."*

논문이 보고한 결과는 다음과 같습니다. 동일 FLOPs/token 기준 T5 대비 **7배의 사전학습 속도 향상**, 1.6T 파라미터(2048 전문가)의 Switch-C 모델로 T5-XXL 대비 4배의 학습 가속을 달성했습니다.

![Switch Transformer Layer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/03_switch_layer.png)
*Switch Transformer 레이어. 각 토큰이 단 하나의 전문가로만 라우팅됩니다. (https://huggingface.co/blog/moe)*

### Load Balancing Loss

Top-1으로 단순화하면서 부하 불균형 문제가 더 두드러집니다. 모든 토큰이 한 전문가로만 몰리면 다른 전문가들은 학습되지 않고 GPU도 유휴 상태로 남습니다. Switch Transformer는 다음 보조 손실로 이를 완화합니다.

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

각 항의 의미는 다음과 같습니다.

- $f_i = \frac{1}{T} \sum_{x \in \mathcal{B}} \mathbb{1}\{\arg\max p(x) = i\}$: 배치 내에서 전문가 $i$에 라우팅된 토큰의 비율
- $P_i = \frac{1}{T} \sum_{x \in \mathcal{B}} p_i(x)$: 배치 내에서 전문가 $i$에 대한 라우터 확률의 평균
- $\alpha = 10^{-2}$: 보조 손실의 가중치 (논문 표준값)

균등 라우팅 상태($f_i = P_i = 1/N$)에서 $\sum_i f_i P_i = 1/N$이 되므로, 앞에 $N$을 곱해 전문가 수에 무관한 스케일을 유지하도록 설계되었습니다. $f_i$는 argmax 지시함수라 미분 불가능하지만, 곱셈 가중치로만 사용되고 그래디언트는 $P_i$를 통해서만 흐르도록 만든 것이 이 손실의 핵심입니다.

## MoE Routing Implementation

라우팅의 의사코드를 살펴보면 동작이 더 명확해집니다.

```python
# Input: x ∈ [T, d_model], T = 배치 내 토큰 수
router_logits = x @ W_r                              # [T, N]
router_probs  = softmax(router_logits, axis=-1)
expert_index  = argmax(router_probs, axis=-1)        # top-1
expert_gate   = max(router_probs,    axis=-1)        # 게이트 값 p_{i*}(x)

# Capacity 계산
capacity = int((T / N) * capacity_factor)

# Capacity 초과 토큰 드롭
mask          = one_hot(expert_index, N)             # [T, N]
position      = cumsum(mask, axis=0) * mask
mask         *= (position < capacity)

# 전문가 연산
expert_inputs  = einsum('tn,td->ntd', mask, x)
expert_outputs = [E_i(expert_inputs[i]) for i in range(N)]
y = expert_gate * scatter(expert_outputs, expert_index)

# 보조 손실 계산
f = mean(mask,         axis=0)
P = mean(router_probs, axis=0)
L_aux = alpha * N * sum(f * P)
```

## Modern MoE Architectures

세 시조 논문(Shazeer 2017, GShard, Switch Transformer)을 거치고 나면 현대 MoE의 설계 공간이 결정됩니다. (1) 전문가 수와 top-k, (2) 공유 전문가(shared expert) 유무, (3) 부하 균형 방법, (4) MoE 레이어 배치가 그 축입니다.

| 모델 | 총 파라미터 | 활성 파라미터 | 전문가 구성 (shared + routed) | Top-K | MoE 레이어 |
|---|---|---|---|---|---|
| Mixtral 8x7B | 46.7B | 12.9B | 0 + 8 | 2 | 32 (전체) |
| Mixtral 8x22B | 141B | 39B | 0 + 8 | 2 | 56 (전체) |
| DeepSeek-V3 | 671B | 37B | 1 + 256 | 8 | 58 / 61 |
| gpt-oss-120b | 116.8B | 5.1B | 0 + 128 | 4 | 36 |
| gpt-oss-20b | 20.9B | 3.6B | 0 + 32 | 4 | 24 |
| Qwen3-235B-A22B | 235B | 22B | 0 + 128 | 8 | 94 |
| Llama 4 Maverick | 400B | 17B | 1 + 128 | 1 routed | alt. dense/MoE |

### Mixtral 8x7B

Mistral AI의 [Mixtral of Experts](https://arxiv.org/abs/2401.04088) 논문은 각 토큰이 47B의 파라미터에 접근할 수 있지만 추론 시에는 13B만 사용한다고 설명합니다. 정확한 수치는 총 46.7B / 활성 12.9B이며, 32개 레이어 전부가 MoE이고 Attention은 GQA(32 Q heads / 8 KV heads)로 구성되어 있습니다.

추론 FLOPs는 13B 규모의 dense 모델과 비슷하지만, 8개 전문가 전부를 추론 시점에 VRAM에 올려두어야 하기 때문에 FP16 기준 약 93.4 GB의 메모리가 필요합니다.

### DeepSeek-V3: Fine-grained + Shared + Aux-loss-free

[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)는 앞서 정리한 MoE 설계 요소들을 한 모델에 집약한 사례입니다. 61개 레이어 중 처음 3개만 dense이고 나머지 58개가 MoE이며, 각 MoE 레이어는 1개의 공유 전문가와 256개의 라우티드 전문가로 구성됩니다. 라우티드 전문가 중 top-8이 활성화됩니다.

수식으로 쓰면 다음과 같습니다.

$$\mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \cdot \text{FFN}_i^{(r)}(\mathbf{u}_t)$$

여기서 $N_s = 1$, $N_r = 256$, $K_r = 8$입니다. 친화도 점수는 $s_{i,t} = \text{Sigmoid}(\mathbf{u}_t^\top \mathbf{e}_i)$로 계산되는데, DeepSeek-V2의 softmax에서 sigmoid로 바뀐 점이 특징입니다.

특히 주목할 부분은 **보조 손실 없는(auxiliary-loss-free) 부하 균형**입니다. 전문가별 바이어스 $b_i$를 도입하지만, top-K 선택에만 쓰고 실제 게이팅 값에는 쓰지 않습니다.

$$g'_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} + b_i \in \text{TopK}(\{s_{j,t} + b_j\}, K_r) \\ 0, & \text{otherwise} \end{cases}$$

$b_i$는 과부하 전문가에 대해 $\gamma$만큼 감소시키고, 저부하 전문가에 대해서는 증가시킵니다(논문 기준 $\gamma = 0.001$). 논문에서는 *"동적 조정을 통해 DeepSeek-V3는 학습 중 부하를 균형 있게 유지하면서, 순수 보조 손실로 유도한 모델보다 더 좋은 성능을 얻었다"*고 보고합니다.

여기에 MLA(Multi-head Latent Attention)로 KV Cache까지 압축하고 FP8 학습을 결합해, 671B 모델을 2.788M H800 GPU 시간(약 558만 달러, GPU 시간당 $2 기준)에 학습시킨 것으로 보고됩니다.

### Trade-offs

MoE는 *연산은 작은 모델 수준, 용량은 큰 모델 수준*을 동시에 얻을 수 있는 설계지만, 그 대가는 **메모리와 통신**으로 돌아옵니다.

| 자원 | $X$B Dense | MoE (총 $X$B, 활성 $Y$B) |
|---|---|---|
| 토큰당 FLOPs | $\propto X$ | $\propto Y$ ($Y < X$) |
| VRAM (가중치) | $\approx X$ | $\approx X$ (모든 전문가가 상주) |
| 통신 | 없음 | All-to-All (디스패치 + 결합) |
| 학습 효율 | 기준 | 같은 perplexity 도달에 2 ~ 7배 적은 시간 |

따라서 MoE는 다중 GPU에서 큰 배치로 추론하는 시나리오에 잘 맞고, 단일 GPU 메모리 제약이 큰 시나리오에서는 상대적으로 불리합니다. 이 한계 때문에 뒤에서 살펴볼 PagedAttention의 페이지드 KV Cache, 전문가 오프로딩(MoE-Infinity, Mixtral-Offloading), MXFP4 같은 후속 작업이 이어지게 됩니다.

---

이제 두 번째 병목인 KV Cache 메모리 낭비 문제로 넘어가보겠습니다.

## The Hidden Waste of KV Cache

이전 글에서 디코드 단계가 Memory-bound이며 KV Cache가 시퀀스 길이에 선형 비례한다고 다뤘습니다. [PagedAttention 논문](https://arxiv.org/abs/2309.06180)(Kwon et al., SOSP 2023)은 여기서 한 발 더 나아가, 실측을 통해 KV Cache 자체가 실제로는 얼마나 비효율적으로 쓰이고 있었는지를 보고했습니다.

> *"Only 20.4% – 38.2% of the KV cache memory is used to store the actual token states in the existing systems."*

기존 시스템들이 KV Cache로 잡아둔 메모리 중 60% 이상이 단편화(fragmentation)로 사용되지 못한 채 남아 있다는 의미입니다.

![Memory Waste](https://blog.vllm.ai/assets/figures/annimation0.gif)
*기존 시스템(왼쪽)과 PagedAttention(오른쪽)의 메모리 사용 비교. (https://blog.vllm.ai)*

### Three Sources of Waste

낭비의 출처는 세 가지로 분해됩니다.

**1. Reserved (예약된 슬롯)**

LLM 추론에서는 출력 시퀀스의 최종 길이를 미리 알 수 없습니다. 그래서 기존 시스템들은 요청이 들어오면 `max_tokens`만큼의 KV Cache 공간을 미리 잡아둡니다. 이 공간은 요청이 끝날 때까지 다른 요청에 쓰일 수 없으며, 실제 생성이 짧게 끝나면 큰 낭비가 됩니다.

**2. Internal Fragmentation**

예약한 공간 중에서 실제로 사용되지 않은 끝부분입니다. 예약 시점에 보수적으로 큰 값을 잡았다가 실제 생성이 짧으면 그 차이만큼 그대로 버려집니다.

**3. External Fragmentation**

서로 다른 크기의 요청이 들어오고 나가는 동안, buddy allocator 같은 메모리 할당자가 블록 사이에 만들어내는 빈 공간입니다.

OPT-13B의 경우, 토큰당 KV 크기는 다음과 같습니다.

$$\underbrace{2}_{K\text{와 }V} \times \underbrace{5120}_{\text{hidden}} \times \underbrace{40}_{\text{layers}} \times \underbrace{2}_{\text{FP16 bytes}} \approx 800\,\text{KB/token}$$

최대 2048 토큰까지 가면 요청 하나당 최대 약 1.6 GB의 KV Cache가 필요합니다. A100 40GB GPU에서 모델 가중치가 약 65%, KV Cache가 약 30%를 차지하는 상황이라면, 그 30% 중 60% 이상이 실제 토큰 저장 용도로 쓰이지 못한다는 뜻이 됩니다.

## PagedAttention: Borrowing from Operating Systems

PagedAttention의 핵심 통찰은 한 문장으로 요약됩니다.

> KV cache blocks are pages, tokens are bytes, requests are processes.

운영체제가 가상 메모리를 페이지 단위로 관리하듯, KV Cache도 고정 크기 블록(기본 $B = 16$ 토큰) 단위로 잘라 물리 메모리의 비연속적인 위치에 분산 배치하고, 블록 테이블(=페이지 테이블)로 논리 블록과 물리 블록을 매핑하는 방식입니다.

![PagedAttention](https://blog.vllm.ai/assets/figures/annimation1.gif)
*PagedAttention의 블록 단위 KV Cache 관리. (https://blog.vllm.ai)*

### Block-level Attention Formulation

원래 Attention은 다음과 같이 정의됩니다.

$$a_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{t=1}^{i} \exp(q_i^\top k_t / \sqrt{d})}, \quad o_i = \sum_{j=1}^{i} a_{ij} v_j$$

PagedAttention은 키와 값을 블록 단위로 묶어 다시 정의합니다.

$$K_j = (k_{(j-1)B+1}, \ldots, k_{jB}), \quad V_j = (v_{(j-1)B+1}, \ldots, v_{jB})$$

그러면 Attention 계산은 다음과 같이 블록 단위로 다시 쓰여집니다.

$$A_{ij} = \frac{\exp(q_i^\top K_j / \sqrt{d})}{\sum_{t=1}^{\lceil i/B \rceil} \exp(q_i^\top K_t \mathbf{1} / \sqrt{d})}, \quad o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^\top$$

여기서 $\mathbf{1}$은 블록 내 합을 만드는 all-ones 벡터입니다. 분자와 분모 모두 한 블록 단위로 끊어서 계산되므로 블록이 물리 메모리에서 연속으로 놓여 있을 필요가 없습니다. GPU 커널은 워프 하나가 블록 하나를 맡아 coalesced read를 수행합니다.

### Memory Allocation Pseudocode

```python
def allocate_for_new_token(seq):
    last = seq.logical_blocks[-1]
    if last.num_filled < B:                       # 현재 블록에 여유가 있으면
        last.num_filled += 1
        write_kv(block_table[seq][last.id], last.num_filled - 1)
    else:                                          # 가득 찼으면 새 물리 블록 할당
        new_id = len(seq.logical_blocks)
        seq.logical_blocks.append(LogicalBlock(num_filled=1))
        phys = block_engine.allocate()
        block_table[seq][new_id] = phys
        write_kv(phys, 0)
```

블록은 왼쪽에서 오른쪽으로 채워지고, 새 물리 블록은 이전 블록이 가득 찼을 때만 할당됩니다. 그 결과 요청당 낭비는 최대 한 블록 이내로 제한됩니다. vLLM 블로그는 실제 낭비를 **4% 미만**으로 보고합니다.

{% include collapse.html title="Memory Waste Comparison Example" %}

OPT-13B에서 평균 시퀀스 길이 256, 최대 2048, 배치 64인 워크로드를 가정해보겠습니다.

**기존 시스템 (정적 예약):**

토큰과 레이어당 KV는 $2 \times 5120 \times 2\,\text{B} = 20\,\text{KB}$이지만, 단순화를 위해 16 KB로 가정합니다.

$$16\,\text{KB} \times 40 \text{ layers} \times 2048 \text{ tokens} \times 64 \text{ batch} \approx 80\,\text{GB}$$

하지만 실제 사용은 평균 시퀀스 길이 기준으로 다음과 같습니다.

$$16\,\text{KB} \times 40 \times 256 \times 64 \approx 10\,\text{GB}$$

> 낭비율: **약 87.5%**

**PagedAttention:**

블록 크기 $B=16$인 경우, 각 시퀀스의 마지막 블록에서만 평균 절반 정도가 낭비됩니다.

$$\text{낭비} \approx \frac{(B/2) \times 16\,\text{KB} \times 40 \times 64}{16\,\text{KB} \times 40 \times 256 \times 64} = \frac{8}{256} \approx 3.1\%$$

추가로 메모리 단편화가 거의 없으므로 전체 낭비는 약 4% 이내에서 유지됩니다.

{% include collapse.html end=true %}

### Copy-on-Write

페이지 단위 KV Cache는 단편화 해소 외에 또 다른 이점을 제공합니다. 바로 **참조 카운트 기반 메모리 공유**입니다. 동일한 프롬프트로 여러 샘플을 생성하는 parallel sampling이나 beam search에서, 모든 후보는 처음에는 같은 물리 블록을 가리키고 참조 카운트가 분기 수만큼 올라갑니다. 한 후보가 블록을 수정해야 할 때만 OS의 `fork()`처럼 Copy-on-Write가 일어납니다.

![Parallel Sampling with PagedAttention](https://blog.vllm.ai/assets/figures/annimation2.gif)
*Parallel sampling에서 여러 출력이 동일한 프롬프트 블록을 공유하는 모습. (https://blog.vllm.ai)*

![Copy-on-Write Block Sharing](https://blog.vllm.ai/assets/figures/annimation3.gif)
*공유된 블록에 새 토큰을 기록할 때 Copy-on-Write가 일어나는 과정. (https://blog.vllm.ai)*

```python
def append_kv_with_cow(seq, logical_id, new_kv):
    phys = block_table[seq][logical_id]
    if ref_count[phys] > 1:                  # 공유 중인 블록은 복사 후 수정
        new_phys = block_engine.allocate()
        fused_block_copy(phys, new_phys)
        ref_count[phys] -= 1
        ref_count[new_phys] = 1
        block_table[seq][logical_id] = new_phys
        phys = new_phys
    write_kv(phys, seq.logical_blocks[logical_id].num_filled)
```

논문의 §6.3에 따르면, OPT-13B와 Alpaca 데이터셋에서 parallel sampling은 6.1 ~ 9.8%의 블록 절약을 보였고, beam search는 37.6 ~ 55.2%의 절약을 달성했습니다. ShareGPT에서는 beam search 절약폭이 최대 66.3%까지 보고됩니다. WMT16 영-독 번역에서는 시스템 프롬프트(341 토큰의 5-shot 예시)를 공유 블록 풀에 캐싱하면 Orca 대비 **3.58배의 처리량 향상**이 보고되었습니다.

## vLLM: The Inference Engine

vLLM은 PagedAttention을 중심으로 다음 컴포넌트를 결합한 추론 엔진입니다.

```
┌─────────────────────────────────────────────────┐
│  Scheduler (Continuous Batching)                │
│   - Iteration 단위로 배치 재구성                │
│   - FCFS 정책, 메모리 부족 시 선점              │
├─────────────────────────────────────────────────┤
│  Centralized KV Cache Manager                   │
│   - Block table 관리 및 ref counting            │
│   - 모든 GPU 워커에 block table 브로드캐스트    │
├─────────────────────────────────────────────────┤
│  Block Engine                                   │
│   - GPU DRAM에 물리 블록 풀 사전 할당           │
│   - CPU RAM에 대칭적인 swap 풀                  │
├─────────────────────────────────────────────────┤
│  PagedAttention Kernel                          │
│   - 블록 단위 coalesced read                    │
│   - FlashAttention 스타일 online softmax        │
└─────────────────────────────────────────────────┘
```

### Continuous Batching

vLLM의 스케줄러는 Orca(OSDI '22)에서 빌려온 **반복(iteration) 단위 스케줄링**을 채택합니다. 매 forward step마다 배치 구성을 바꿔서, 종료된 시퀀스는 즉시 빠지고 새 요청은 즉시 합류합니다. 입출력 패딩이 없으므로 GPU 유휴 시간이 크게 줄어듭니다.

### Preemption Strategies

메모리가 부족하면 가장 늦게 들어온 요청을 선점합니다. 두 가지 전략이 있습니다.

- **Swap**: KV 블록을 CPU RAM으로 옮겨둠
- **Recompute**: KV를 버리고 나중에 재계산

논문 §7.3에 따르면 블록 크기 16 ~ 64에서 두 방식의 성능은 비슷하고, 재계산 오버헤드는 스왑의 20%를 넘지 않습니다.

### Throughput Benchmarks

논문 초록의 핵심 클레임은 다음과 같습니다.

> *"vLLM improves the throughput of popular LLMs by 2-4× compared to state-of-the-art systems, such as FasterTransformer and Orca, at the same level of latency."*

세부 수치를 보면 다음과 같습니다.

- ShareGPT / OPT-13B 기본 샘플링: Orca(Oracle) 대비 1.7 ~ 2.7배, Orca(Max) 대비 2.7 ~ 8배 처리량
- 같은 워크로드에서 vLLM은 Orca(Oracle)보다 2.2배 많은 요청을 동시에 배치
- FasterTransformer 대비 최대 22배
- vLLM 0.1 블로그(LLaMA-7B/13B, ShareGPT): HF Transformers 대비 14 ~ 24배, HF TGI 대비 2.2 ~ 3.5배

![vLLM Throughput on A100](https://blog.vllm.ai/assets/figures/perf_a100_n1_light.png)
*LLaMA-13B / A100 환경에서 단일 출력 시 vLLM과 HF, TGI의 처리량 비교. (https://blog.vllm.ai)*

### The Trade-off

PagedAttention의 성능이 공짜로 얻어지는 것은 아닙니다. PagedAttention 커널 자체는 FasterTransformer의 융합 어텐션보다 20 ~ 26% 느립니다(논문 §7.1). 블록 테이블 접근과 비연속 메모리 접근에서 발생하는 비용입니다. 다만 메모리 활용도가 높아지면서 더 큰 배치를 동시에 처리할 수 있게 되고, 그 결과 종단(end-to-end) 처리량은 커널 단위 손해를 상쇄하고도 큰 폭으로 앞서게 됩니다.

## Combination in Modern LLMs

이전 글과 이 글에서 다룬 기법들은 서로 다른 축을 공략하기 때문에 함께 적용할 수 있고, 실제 프로덕션 환경에서는 대부분 조합되어 사용됩니다. 각 기법이 공략하는 축을 정리하면 다음과 같습니다.

| 기법 | 공략 지점 | 핵심 원리 | 주요 효과 |
|---|---|---|---|
| **GQA** | KV Cache 크기 | 그룹별 K, V 공유 | 메모리 $H/G$배 감소 |
| **FlashAttention** | HBM IO 횟수 | Tiling + Online Softmax | 어텐션 2 ~ 4배 가속 |
| **MoE** | FFN 활성 파라미터 | Top-K sparse 활성화 | 같은 연산으로 큰 용량 |
| **PagedAttention** | KV 메모리 단편화 | 페이지 단위 비연속 할당 | 종단 처리량 2 ~ 24배 |
| **Continuous Batching** | GPU 유휴 시간 | Iteration 단위 스케줄링 | 패딩 제거, 2 ~ 10배 |

이 기법들이 한 모델 안에 어떻게 함께 쌓이는지 보여주는 대표적인 예가 DeepSeek-V3입니다.

| 구성 요소 | 적용 기법 | 효과 |
|---|---|---|
| Attention | MLA (Multi-head Latent Attention) | KV Cache 약 10배 압축, 128k 컨텍스트 가능 |
| FFN | DeepSeekMoE (1 shared + 256 routed, top-8) | 671B 용량 → 토큰당 37B 활성화 |
| 부하 균형 | Aux-loss-free bias update | 라우터 품질 + 균형 동시 확보 |
| 학습 정밀도 | FP8 | 학습 비용 ~1/10 |
| 서빙 | vLLM/SGLang의 PagedAttention + EP All-to-All | NVLink/IB로 통신 오버랩 |

OpenAI의 gpt-oss-120b도 비슷한 패턴을 따릅니다.

| 구성 요소 | 적용 기법 | 효과 |
|---|---|---|
| Attention | GQA ($H=64$, $H_{kv}=8$) | KV Cache 8배 절감 |
| Attention 레이어 | Full + 128-token Sliding Window 교대 | 긴 컨텍스트 효율화 |
| FFN | MoE (experts=128, top-4) | 117B 중 토큰당 5.1B 활성화 |
| 가중치 양자화 | MXFP4 native 학습 | H100 80GB 한 장에 적재 |
| 서빙 | vLLM (PagedAttention) | 표준 추론 스택 |

## What Became the Next Standard (2026 Update)

이 글의 초고를 작성할 시점에 다음 후보로 거론되던 기법들은 1년 사이에 일부는 프로덕션 표준에 가까워졌고, 일부는 그렇게 되지 못했습니다.

### Prefill-Decode Disaggregation

이 중 가장 분명하게 자리 잡은 기법입니다. Compute-bound인 Prefill 단계와 Memory-bound인 Decode 단계의 자원 요구가 서로 달라, 같은 GPU에 묶어두면 양쪽 모두에서 자원 활용이 비효율적이 됩니다. UCSD Hao AI Lab의 회고에 따르면 2025년 말 기준으로 NVIDIA Dynamo, llm-d, Ray Serve LLM, SGLang, vLLM, LMCache, Mooncake 등 다수의 프로덕션급 서빙 프레임워크가 disaggregation을 기반으로 동작합니다. Meta, LinkedIn, Mistral, Hugging Face가 이미 프로덕션에 도입했고, 보고된 효과는 워크로드에 따라 처리량 70% 향상과 TTFT 90% 단축 수준입니다.

### NVIDIA Dynamo

2026년 3월 정식 출시된 [NVIDIA Dynamo 1.0](https://developer.nvidia.com/dynamo)은 vLLM/SGLang/TensorRT-LLM 같은 엔진 위에 올라가는 오케스트레이션 레이어입니다. AWS, Azure, GCP, OCI 같은 하이퍼스케일러와 Perplexity, Cursor, Baseten, ByteDance, PayPal, Pinterest 등이 채택했습니다. 핵심 컴포넌트는 다음과 같습니다.

- **KVBM (KV Block Manager)**: GPU HBM → CPU RAM → SSD → 네트워크 스토리지로 KV Cache 계층 오프로딩
- **NIXL**: RDMA 기반 GPU-to-GPU KV 전송 라이브러리
- **Grove**: Kubernetes 위의 hierarchical gang scheduling
- **LLM-aware Routing**: 동일 prefix 요청을 같은 노드로 라우팅해 KV 재계산 회피

### NVFP4 / MXFP4 Quantization

Blackwell B200/B300이 FP4를 하드웨어 네이티브로 지원하면서 4비트 양자화가 프로덕션 영역에 들어왔습니다. NVFP4는 MXFP4를 확장한 것으로, 블록 크기를 32에서 16으로 줄이고 two-level scaling(per-block E4M3 + per-tensor FP32)을 도입해 FP8 대비 2 ~ 3배의 산술 처리량과 약 1.8배의 메모리 감소를 제공합니다. NVIDIA가 `nvidia/Llama-3.3-70B-Instruct-NVFP4`, `nvidia/DeepSeek-R1-NVFP4` 같은 공식 체크포인트를 배포하고, vLLM/TensorRT-LLM이 네이티브로 디코딩합니다.

### Speculative Decoding (EAGLE-3, MTP)

2025 ~ 2026년 사이에 주요 서빙 프레임워크의 기본 옵션 중 하나로 자리 잡았습니다. vLLM, SGLang, TensorRT-LLM 모두 네이티브로 지원하며, 프로덕션 벤치마크는 낮은-중간 동시성 구간에서 2.0 ~ 6.5배의 처리량 향상을 보고합니다. DeepSeek-V3에 내장된 MTP(Multi-Token Prediction)는 80% 이상의 acceptance rate 조건에서 약 1.8배의 가속이 보고되었습니다.

### 표준 자리에 도달하지 못한 후보들

반대로 표준 자리에 도달하지 못한 기법들도 있습니다.

- **vAttention**: 설계 자체는 깔끔하지만, vLLM v1의 prefix caching + LMCache 조합이 먼저 자리 잡으면서 채택 동력이 약해졌습니다.
- **Mamba-Transformer 하이브리드**: Jamba(AI21), Nemotron-H(NVIDIA), Falcon-Mamba 같은 모델들이 등장했지만, 프런티어 모델 다수는 여전히 Transformer-MoE 스택을 채택하고 있습니다.
- **Expert Offloading**: 단일 GPU 시나리오에서 주로 의미가 있고, 프로덕션은 EP(Expert Parallelism)로 다중 GPU에 분산하는 방향으로 정착했습니다.

## Conclusion

LLM 추론 최적화는 단일 기법으로 완성되지 않습니다. 이전 글의 GQA와 FlashAttention이 Attention 측 비용을 줄였다면, 이 글에서 다룬 두 기법은 그 다음에 남는 두 병목(FFN 연산과 KV Cache 메모리 사용)을 각각 공략합니다.

| 기법 | 공략 지점 | 핵심 원리 | 주요 효과 |
|---|---|---|---|
| **MoE** | FFN 연산량 | Top-K sparse 활성화 + 전문가 복제 | 같은 연산량으로 더 큰 모델 용량 확보 |
| **PagedAttention** | KV Cache 메모리 단편화 | 페이지 테이블 + 비연속 할당 | KV 메모리 낭비 60%+ → 4% 이하, 처리량 2 ~ 24배 |

MoE는 FFN을 sparse하게 만들어 용량은 크고 토큰당 연산은 작은 모델 설계를 가능하게 했습니다. Shazeer의 Sparsely-Gated MoE에서 시작해, GShard의 Top-2와 Expert Capacity, Switch Transformer의 Top-1과 단순한 부하 균형 손실, DeepSeek-V3의 보조 손실 없는 바이어스 갱신으로 이어지는 흐름이 있었습니다. Mixtral 8x7B의 47B/13B, DeepSeek-V3의 671B/37B, gpt-oss-120b의 117B/5.1B 같은 현대 프런티어 모델들이 이 흐름 위에 서 있습니다.

PagedAttention은 운영체제의 페이징을 가져와 KV Cache의 단편화 낭비를 4% 미만 수준으로 끌어내렸습니다. 여기에 Copy-on-Write 기반 블록 공유로 parallel sampling, beam search, 시스템 프롬프트 공유까지 함께 가속할 수 있게 되었고, 이를 기반으로 한 vLLM은 현재 가장 널리 사용되는 오픈소스 추론 엔진 중 하나로 자리 잡았습니다.

두 기법은 서로 직교하는 축을 다룹니다. MoE는 모델 아키텍처 안에서, PagedAttention은 서빙 시스템 안에서 작동하며, 현대 프로덕션 스택은 이전 글에서 다룬 GQA/MLA + FlashAttention과 이 글에서 다룬 MoE + PagedAttention + Continuous Batching을 함께 쌓아 올립니다. 2026년 현재 그 위에 다시 **Prefill-Decode Disaggregation**과 **NVIDIA Dynamo의 오케스트레이션 레이어**, 그리고 **NVFP4 양자화**와 **Speculative Decoding**이 새로운 표준으로 자리를 잡고 있습니다. 다음 글에서는 이 새 표준 — 특히 disaggregated serving의 아키텍처를 KV Cache 전송 비용과 함께 다뤄보겠습니다.
