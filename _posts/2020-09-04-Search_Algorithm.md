---
date: 2020-09-04 18:30:28
layout: post
title: Linear Search & Binary Search
subtitle: Research
# description: Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
math: true
image: https://res.cloudinary.com/dm7h7e8xj/image/upload/v1559824575/theme14_gi2ypv.jpg
optimized_image: https://res.cloudinary.com/dm7h7e8xj/image/upload/c_scale,w_380/v1559824575/theme14_gi2ypv.jpg
category: Algorithm
tags:
    - Algorithm
    - Problem Solving
    - Linear Search
    - Binary Search
author: pyy0715
---

이 글은 Edwith에서 제공하는 [모두를 위한 컴퓨터 과학 (CS50 2019)](https://www.edwith.org/boostcourse-cs-050/joinLectures/41307)를 수강하고 정리한 글입니다.

배열은 한 자료형의 여러 값들이 메모리상에 모여 있는 구조입니다.
컴퓨터는 이 값들에 접근할 때 배열의 인덱스 하나하나를 접근합니다.
만약 어떤 값이 배열 안에 속해 있는지를 찾아 보기 위해서는 배열이 정렬되어 있는지 여부에 따라 아래와 같은 방법을 사용할 수 있습니다.

# Linear Search

찾고자 하는 자료를 검색하는 데 사용되는 다양한 알고리즘이 있습니다. 그 중 하나가 선형 검색입니다.
선형검색은 원하는 원소가 발견될 때까지 처음부터 마지막 자료까지 차례대로 검색합니다.
이렇게 하여 선형 검색은 찾고자 하는 자료를 찾을 때까지 모든 자료를 확인해야 합니다.

## 효율성 그리고 비효율성

선형 검색 알고리즘은 정확하지만 아주 효율적이지 못한 방법입니다.
리스트의 길이가 $n$이라고 했을 때, 최악의 경우 리스트의 모든 원소를 확인해야 하므로 $n$번만큼 실행됩니다.
따라서 입력 데이터의 크기에 비례해 처리 시간이 증가하기 때문에 시간복잡도는 $O(n)$이 됩니다.

여기서 최악의 상황은 찾고자 하는 자료가 맨 마지막에 있거나 리스트 안에 없는 경우를 말합니다.
만약 100만 개의 원소가 있는 리스트라고 가정해본다면 효율성이 매우 떨어짐을 느낄 수 있습니다.
반대로 최선의 상황은 처음 시도했을 때 찾고자 하는 값이 있는 경우입니다.

평균적으로 선형 검색이 최악의 상황에서 종료되는 것에 가깝다고 가정할 수 있습니다.
선형 검색은 자료가 정렬되어 있지 않거나 그 어떤 정보도 없어 하나씩 찾아야 하는 경우에 유용합니다.
이러한 경우 무작위로 탐색하는 것보다 순서대로 탐색하는 것이 더 효율적입니다.

이제 여러분은 왜 검색 이전에 정렬해줘야 하는지 알 수 있을 것입니다.
정렬은 시간이 오래 걸리고 공간을 더 차지합니다.
하지만 이 추가적인 과정을 진행하면 여러분이 여러 번 리스트를 검색해야 하거나 매우 큰 리스트를 검색해야 할 경우 시간을 단축할 수 있을 것입니다.

주어진 배열에서 특정 값을 찾기 위해서 선형 검색을 사용한다면, 아래와 같은 코드를 작성할 수 있습니다.

```python
def Linear_Search(arr, target):
   """
   Linear Search Alogirhm
   Args:
      arr: input array
      target: target
   Return:
      target index
   """
   for idx, i in enumerate(arr):
      if i==target:
         print(f'Found Target: {idx}')
         return idx 
         break
   return 'Not Found'
``` 

# Binary Search

이진 탐색 알고리즘은 오름차순으로 정렬된 리스트에서 특정한 값의 위치를 찾는 알고리즘입니다.

만약 이미 배열이 정렬되어 있다면, 배열의 중간 인덱스부터 시작하여 찾고자 하는 값과 비교하며, 그보다 작은(작은 값이 저장되어 있는) 인덱스 또는 큰 (큰 값이 저장되어 있는) 인덱스로 이동을 반복하면 됩니다.

아래 코드와 같이 나타낼 수 있습니다.

```python
def Binary_Search(arr, target):
   """
   Binary Search Alogirhm
   Args:
      arr: sorted input array
      target: target
   Return:
      target index
   """
   n = len(arr)
   begin = 0
   end = n-1
   idx = -1
   while begin<=end:
      mid = (begin+end)//2
      if arr[mid]<=target:
         begin = mid + 1
         idx = mid 
      else:
         end = mid -1
   return idx
``` 

## Time Complexity

![image](https://user-images.githubusercontent.com/47301926/92421233-810ff200-f1b2-11ea-95be-54e0d5954e6a.png)

위의 그림과 같이 $n$개의 배열이 있을때, 이진 탐색 알고리즘은 Target을 찾기 위해 매번 탐색 범위를 $n/2$로 줄이게 됩니다.
따라서 연산의 횟수를 $k$번이라고 하였을떄, $n \times\left(\frac{1}{2}\right)^{k}=1$ 라 할 수 있습니다.

위 식을 다시 k에 대해 정리하면 다음과 같습니다.

$$ 2^k = n $$

$$ k=\log _{2} n $$

따라서 시간 복잡도는 Big O 표기법으로 $O(log N)$이 됩니다.

# Next

* [Algorithm] Bubble Sort & Selection Sort

* [Algorithm] Merge Sort