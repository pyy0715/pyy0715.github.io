---
date: 2021-04-07 18:39:28
layout: post
title: 메소드와 데코레이터
subtitle: Research
# description: Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
math: true
image: https://images.unsplash.com/photo-1617201835175-aab7b1d71d87?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=2765&q=80
optimized_image: https://images.unsplash.com/photo-1617201835175-aab7b1d71d87?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=2765&q=80
category: Python
tags:
    - Decorator
    - Method
author: pyy0715
---

> **진지한 파이썬** 7장의 내용을 공부하면서 정리한 내용입니다.

메서드와 데코레이터를 공부하기 전에 헷갈릴 수 있는 일부 용어들을 정리할 필요가 있습니다.

# 객체와 인스턴스의 차이

> 클래스로 만든 객체를 인스턴스라고도 한다. 그렇다면 객체와 인스턴스의 차이는 무엇일까? 이렇게 생각해 보자. a = Cookie() 이렇게 만든 a는 객체이다. 그리고 a 객체는 Cookie의 인스턴스이다. 즉 인스턴스라는 말은 특정 객체(a)가 어떤 클래스(Cookie)의 객체인지를 관계 위주로 설명할 때 사용한다. "a는 인스턴스"보다는 "a는 객체"라는 표현이 어울리며 "a는 Cookie의 객체"보다는 "a는 Cookie의 인스턴스"라는 표현이 훨씬 잘 어울린다.

*by 점프 투 파이썬*

# 데코레이터란?

파이썬에서 decorator를 사용하면 함수를 편리하게 수정할 수 있습니다. decorator는 classmethod()와 staticmethod()와 함께 파이썬 2.2에서 처음 도입되었고, 유연성과 신뢰성을 높이는 방향으로 개선되어왔습니다.

데코레이터는 다른 함수를 인수로 받아 새롭게 수정된 함수로 대체하는 방법을 뜻하며 기본적인 사례는 호출해야하는 공통 코드를 리팩터링 하는 것입니다.

아래의 예제처럼 `@기호` 뒤에 데코레이터의 이름을 입력한 다음 사용할 함수를 입력합니다.
```python
def identity(f):
   return f

@identity
def foo():
   return 'bar'
```

데코레이터는 함수를 중심으로 반복되는 코드를 리팩토링할 때 자주 사용됩니다.  아래의 예제에서 코드를 좀 더 효율적으로 만들기 위해서는 관리자 상태를 확인하는 코드를 리팩토링하는 것입니다.

```python
class Store(object):
    def get_food(self, username, food):
        if username != 'admin':
            raise Exception('This user is not allowed to get food')
        return self.storage.get(food)

    def put_food(self, username, food):
        if username != 'admin':
            raise Exception('This user is not allowed to get food')
        return self.storage.put(food)
```

```python
#  조금은 더 효율적인 코드
def check_is_admin(username):
    if username != 'admin':
        raise Exception('This user is not allowed to get food')

class Store(object):
    def get_food(self, username, food):
        check_is_admin(username)
        return self.storage.get(food)

    def put_food(self, username, food):
				check_is_admin(username)
        return self.storage.put(food)
```

유저가 관리자인지를 확인하는 검사 코드를 자체 함수로 이동시킴으로써, 코드가 좀 더 간결해졌지만 데코레이터를 사용하면 좀 더 간단하게 만들 수 있습니다.

```python
# 데코레이터를 사용하여 훨씬 간결한 코드
def check_is_admin(f):
    def wrapper(*args, **kwargs):
        if kwargs.get('username') != 'admin':
            raise Exception('This user is not allowed to get or put food')
        return f(*args, **kwargs)
    return wrapper

class Store(object):
    @check_is_admin 
    def get_food(self, username, food):
        return self.storage.get(food)

    @check_is_admin 
    def put_food(self, username, food):
        return self.storage.put(food)
```

데코레이터는 kwargs 변수를 사용하여 함수에 전달된 인수를 검사하고 username인수를 검색합니다. 그리고 실제 함수를 호출하기 전에 사용자 이름 검사를 수행합니다.

## 여러 데코레이터 사용하기

단일 함수 또는 메서드 위에 여러 데코레이터를 사용할 수도 있습니다. 다음은 단일 함수를 사용하여 하나 이상의 데코레이터를 사용하는 예제입니다.

```python
def check_user_is_not(username):
    def user_check_decorator(f):
        def wrapper(*args, **kwargs):
            if kwargs.get('username') != username:
                raise Exception('This user is not allowed to get or put food')
            return f(*args, **kwargs)
        return wrapper
    return user_check_decorator

class Store(object):
    @check_user_is_not('admin') 
    @check_user_is_not('user123') 
    def get_food(self, username, food):
        return self.storage.get(food)
```

`check_user_is_not` 는 `user_check_decorator` 에 대한 팩터리 함수입니다. username 변수에 종속된 함수 데코레이터를 만들고 해당 변수를 반환합니다. `user_check_decorator` 는 `get_food()`에 대한 함수 데코레이터 역할을 합니다.

즉 `get_food()`는 `check_user_is_not()`를 사용하여 두 번 데코레이터 되어집니다.

데코레이터 목록은 위에서 아래로 적용되므로 def 키워드에 가장 가까운 데코레이터가 먼저 적용되고 마지막으로 실행됩니다. 위의 예제에서 프로그램은 admin을 먼저 확인한 다음 user123을 확인하게 됩니다.

## wraps: 데코레이터용 데코레이터

앞서 말했듯이 데코레이터는 원래의 함수를 제작하여 새로운 함수로 대체합니다. 그러나 함수에 대한 올바른 docstring과 이름 속성이 없으면, 소스코드를 문서화할 때의 같이 다양한 상황에서 문제가 발생할 수 있습니다.

다음은 함수 foobar()가 `check_is_admin`데코레이터가 되면서 docstring과 이름에 대한 속성을 잃는 방법을 보여줍니다.

```python
def check_is_admin(f):
    def wrapper(*args, **kwargs):
        if kwargs.get('username') != admin:
            raise Exception('This user is not allowed to get or put food')
        return f(*args, **kwargs)
    return wrapper

@check_is_admin
def foobar(username='someone'):
    """구현할 매서드 내용"""
    pass

print(foobar.__doc__) # None
print(foobar.__name__) # wrapper
```

파이썬 표준 라이브러리의 functools 모듈은 래퍼 자체에 손실된 원래 함수의 속성을 복사하는 데코레이터를 위한 데코레이터로 이 문제를 해결합니다.

`functools.wrap`을 사용하면 wrapper()함수를 반환하는 데코레이터 함수 `check_is_admin()`은 인수로 전달된 함수 f에서 docstring, 함수 이름, 기타 정보를 복사합니다.

```python
import functools 

def check_is_admin(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if kwargs.get('username') != admin:
            raise Exception('This user is not allowed to get or put food')
        return f(*args, **kwargs)
    return wrapper

@check_is_admin
def foobar(username='someone'):
    """구현할 매서드 내용"""
    pass

print(foobar.__doc__) # 구현할 매서드 내용
print(foobar.__name__) # foobar
```

이제 foobar()함수는 `check_is_admin` 으로 데코레이팅 할 떄도 올바른 이름과 docstring을 사용합니다.

# 파이썬에서 매서드가 작동하는 방법

매서드는 사용하고 이해하기가 매우 간단합니다. 그러나 특정 데코레이터가 하는 일을 이해하려면 매서드가 실제로 어떻게 작동하는지를 알아야 합니다.

```python
class Pizza(object):
    def __init__(self, size):
        self.size = size 
    def get_size(self):
        return self.size

print(Pizza.get_size) # <function Pizza.get_size at 0x7febe036d8c8>
print(Pizza.get_size()) # TypeError: get_size() missing 1 required positional argument: 'self'
print(Pizza.get_size(Pizza(42)))
```

우리는 `get_size()`가 어떤 특정 객체에 연결되지 않았기 때문에 정상적인 함수인 것을 알고있지만, 호출 시 오류를 발생합니다. 이는 함수가 객체에 구속되지 않기 때문에 자체 인수를 자동으로 설정할 수 없기 떄문입니다.

따라서 클래스의 임의 인스턴스를 매서드에 전달하여 함수를 사용할 수 있습니다. 하지만 이러한 방법은 매서드를 호출할 때마다 클래스를 참조해야 하기 때문에 편리하지 않습니다.

파이썬은 클래스의 메서드를 인스턴스에 바인딩하여 많은 정보를 제공합니다. pizza 인스턴스에서 `get_size()`에 엑세스할 수 있으며, 자동으로 객체 자체를 메서드의 자체 매개변수로 전달하는 것입니다.

```python
print(Pizza(42).get_size) # <bound method Pizza.get_size of <__main__.Pizza object at 0x7fdc40560b00>>
print(Pizza(42).get_size()) # 42

m = Pizza(42).get_size
print(m()) # 42
print(m.__self__) # <__main__.Pizza object at 0x7fdc40560b00>
print(m==m.__self__.get_size) # True
```

## 정적 메서드

정적 메서드는 클래스의 인스턴스가 아니라 클래스에 속하므로 실제로 클래스 인스턴스에서 작동하거나 영향을 주지 않습니다. 대신 정적 메서드는 가지고 있는 매개변수에서 작동합니다. 정적 메서드는 일반적으로 클래스 또는 해당 객체의 상태에 의존하지 않기 때문에 유틸리티 함수를 만드는데 사용합니다.


```python
class Pizza(object):
    @staticmethod
    def mix_ingredients(x, y):
        return x + y

    def cook(self):
        return selfz.mix_ingredients(self.cheese, self.vegetables)
```
정적 `mix_ingredients()` 메서드는 Pizza 클래스에 속하지만, 클래스와 상관없이 메서드를 사용할 수 있습니다.


`@staticmethod` 를 사용하면 다음과 같은 기능을 제공합니다.

- 우리가 만드는 각 Pizza객체에 대한 바인딩된 메서드를 인스턴스화 할 필요가 없기 때문에 속도 측면에서 유리합니다.
- 메서드가 객체의 상태에 의존하지 않는다는 것을 알 수 있기 때문에 코드의 가독성이 향상됩니다.
- 정적 메서드는 하위 클래스에서 재정의할 수 있습니다. 하지만 메서드가 정적인지 아닌지 항상 자체적으로 감지할 수는 없기 때문에 flake8을 사용하여 패턴을 감지하고 경고를 발생하는 검사를 수행할 수 있습니다.

## 클래스 메서드

클래스 메서드는 인스턴스가 아닌 클래스에 바인딩됩니다. 즉 이러한 메서드는 객체의 상태에 엑세스 할 수 없고 클래스의 상태 및 메서드만 엑세스 할 수 있습니다.

```python
class Pizza(object):
    radius = 42
    @classmethod
    def get_radius(cls):
        return cls.radius

print(Pizza.get_radius) #<bound method Pizza.get_radius of <class '__main__.Pizza'>>
print(Pizza().get_radius) #<bound method Pizza.get_radius of <class '__main__.Pizza'>>
print(Pizza.get_radius is Pizza().get_radius()) #False
print(Pizza.get_radius()) #42
```

위의 예제처럼 get_radius() 클래스 메서드에 엑세스하는 다양한 방법이 있지만 메서드는 항상 연결된 클래스에 바인딩됩니다. 

클래스 메서드는 주로  `__init__` 보다 다른 서명을 사용하여 객체를 인스턴스화하는 팩터리 메서드를 만드는데 사용됩니다.

```python
class Pizza(object):
    def __init__(self, ingredients):
        self.ingredients = ingredients

    @classmethod
    def from_fridge(cls, fridge):
        return cls(fridge.get_cheese() + fridge.get_vegetables())
```

만약에 위의 예제어서  `@classmethod` 대신에 `@staticmethod` 를 사용하면 Pizza에서 상속받은 클래스가 자체 목적으로 팩터리를 사용할 수 없게 되어 Pizza클래스 이름을 하드코딩해야 할 것입니다.

즉 객체의 상태에 대한 것이 아니라 객체의 클래스에만 관심이 있는 메서드를 작성할 때마다 클래스 메서드로 선언해야 합니다.

## 추상 메서드

추상 메서드는 구현 자체를 제공하지 않을 수 있는 추상 기본 클래스에서 정의됩니다. 클래스에 추상 메서드가 있다면 인스턴스화 할 수 없습니다. 따라서 추상 클래스를 다른 클래스에서 부모 클래스에서 사용해야 합니다.

추상 기본 클래스를 사용하여 파생된 다른 연결된 클래스 간의 관계를 명확히 할 수 있지만, 추상 기본 클래스 자체를 인스턴스화하는 것은 불가능합니다. 

추상 기본 클래스를 사용하여 기본 클래스에서 파생된 클래스는 기본 클래스에서 특정 메서드를 구현하도록 보장하거나 예외를 발생시킵니다.

```python
import abc

class BasePizza(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_radius(self):
        """구현할 메서드 내용"""

print(BasePizza() #TypeError: Can't instantiate abstract class BasePizza with abstract method get_radius
```

추상 BasePizza 클래스를 인스턴스화 하려고 하면, 즉시 할 수 없다는 피드백을 받습니다.

추상 메서드를 사용한다고 해서 사용자가 메서드를 구현하는 것이 보장되지는 않지만 이 데코레이터는 오류를 더 일찍 잡는 데 도움이 되거나 다른 개발자가 구현해야 하는 인터페이스를 제공화 할 때 특히 유용합니다.