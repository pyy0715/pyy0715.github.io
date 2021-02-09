---
date: 2021-02-09 18:39:28
layout: post
title: HuggingFace Tokenizer Tutorial
subtitle: Research
# description: Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
math: true
image: https://res.cloudinary.com/dm7h7e8xj/image/upload/v1559824822/theme15_oqsl4z.jpg
optimized_image: https://res.cloudinary.com/dm7h7e8xj/image/upload/c_scale,w_380/v1559824822/theme15_oqsl4z.jpg
category: NLP
tags:
    - HuggingFace
    - Tokenizer 
author: pyy0715
---
# HuggingFace

![img](https://pbs.twimg.com/media/EUngU6UXQAISNL_.jpg:large)
지난 2년간은 NLP에서 황금기라 불리울 만큼 많은 발전이 있었습니다. 그 과정에서 오픈 소스에 가장 크게 기여한 곳은 바로 [HuggingFace](https://huggingface.co/)라는 회사입니다. 
HuggingFace는 Transformer, Bert등의 최신 NLP 기술들을 많은 이들이 쉅게 사용할 수 있도록 기술의 민주화를 목표로 하고 있습니다. 
> We’re on a journey to advance and democratize NLP for everyone. Along the way, we contribute to the development of technology for the better.

이번 포스트에는 HuggingFace에서 제공하는 [Tokenizers](https://github.com/huggingface/tokenizers)를 통해 각 기능을 살펴보겠습니다.

## What is Tokenizer?

우선 Token, Tokenizer 같은 단어들에 혼동을 피하기 위해서 의미를 정리할 필요가 있습니다.
- **Token**은 주어진 Corpus에서 의미있는 단위로 정의되는 문자로 정의할 수 있습니다.
    의미있는 단위란 문장, 단어나 어절 등이 될 수 있습니다.

- **Tokenizer**은 주어진 Corpus를 기준에 맞춰서 Token들로 분리하는 작업을 뜻합니다.
    기준은 사용자가 지정하거나 사전에 기반하여 정할 수 있습니다.
    이러한 기준은 *사전 기반과 Subword기반* 으로 구분될 수 있으며, 각자 목적에 맞게 사용됩니다.


# Tokenizers Introduction
- 오늘날 가장 많이 사용되는 Tokenizer를 사용하여 새로운 어휘를 훈련하고, Tokenize를 수행할 수 있습니다.
- Rust로 구현되있기 때문에 매우 빠릅니다.
- 사용하기 쉬우면서도 매우 다재다능합니다.
- 연구 및 생산을 위해 설계되었습니다.
- 주어진 토큰에 해당하는 원래 문장의 일부를 항상 가져올 수 있습니다.
- 전처리에 관한 모든것을 수행할 수 있습니다(Truncate, Pad, add the special tokens)

# Kind of Tokenizers

|           Tokenizer(class name)           | Unit |  Method  |              Normalizer              |    Symbol    |
|:-----------------------------------------:|:----:|:---------:|:------------------------------------:|:---------------------:|
|  Bert tokenizer<br />(BertWordPieceTokenizer)  | char | WordPiece |            BertNormalizer            | subword 앞에 `##` 부착|
| SentencePiece<br />(SentencePieceBPETokenizer) | char |    BPE    |                 NFKC                 |       어절 앞 `_`     |
|   Byte-level BPE<br />(ByteLeveBPETokenizer)   | byte |    BPE    |         [Unicode, Lowercase]         |       어절 앞 `Ġ`     |
|   Character-level BPE<br />(CharBPETokenizer)  | char |    BPE    | [Unicode, BertNormalizer, Lowercase] |      어절 뒤 `</w>`   |

# Code Pratice

## Preparations

튜토리얼을 진행하기 앞서, 간단한 문장들로 구성된 텍스트 파일을 생성하겠습니다.
```python
sentences = (
    'Joe waited for the train.',
    'The train was late.',
    'Mary and Samantha took the bus.',
    'I looked for Mary and Samantha at the bus station',
)
```

```python
with open('sample_corpus.txt', 'w') as f:
    for data in sentences:
        f.write(data+'\n')
```

## BertWordPieceTokenizer
WordPiece는 BPE와 같이 가장 많이 등장한 쌍을 병합하는 것이 아니라, 병합되었을 때 corpus의 우도를 가장 높이는 쌍을 병합하게 됩니다.

```python
from tokenizers import BertWordPieceTokenizer

bert_wordpiece_tokenizer = BertWordPieceTokenizer(lowercase=True)

bert_wordpiece_tokenizer.train(
    files = './sample_corpus.txt',
    vocab_size = 30,
    min_frequency = 1,
    limit_alphabet = 1000,
    initial_alphabet = [],
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    show_progress = True,
    wordpieces_prefix = "##",
)
```

`min_frequency` : merge를 수행할 최소 빈도수, 5로 설정 시 5회 이상 등장한 pair만 수행한다

`vocab_size`: 만들고자 하는 vocab의 size

`show_progress` : 학습 진행과정 show

`special_tokens` : Tokenizer에 추가하고 싶은 special token 지정

`limit_alphabet` : merge 수행 전 initial tokens이 유지되는 숫자 제한

`initial_alphabet` : 꼭 포함됐으면 하는 initial alphabet, 이곳에 설정한 token은 학습되지 않고 그대로 포함되도록 설정된다.


```python
vocab = bert_wordpiece_tokenizer.get_vocab()
sorted(vocab, key=lambda x: vocab[x])
```

    ['[PAD]',
     '[UNK]',
     '[CLS]',
     '[SEP]',
     '[MASK]',
     '.',
     'a',
     'b',
     'd',
     'e',
     'f',
     'h',
     'i',
     'j',
     'k',
     'l',
     'm',
     'n',
     'o',
     'r',
     's',
     't',
     'u',
     'w',
     'y',
     '##a',
     '##t',
     '##e',
     '##i',
     '##o',
     '##n',
     '##r',
     '##y',
     '##h',
     '##d',
     '##u',
     '##s',
     '##m',
     '##k']

사전에 위에서 추가한 special token들이 잘 포함되있는 것을 확인할 수 있습니다.


### Encode and Encode_Batch


```python
encoding = bert_wordpiece_tokenizer.encode('I take the bus')
print(encoding.tokens)
print(encoding.ids)
```

    ['i', 't', '##a', '##k', '##e', 't', '##h', '##e', 'b', '##u', '##s']
    [12, 21, 25, 38, 27, 21, 33, 27, 7, 35, 36]

위와 같은 하나의 문장을 Encoding하는 것은 `encode` method를 통해서 확인할 수 있지만, 수백개의 문장을 Encoding해야 한다면 어떻게 수행할 수 있을까요?

`encode_batch` method를 사용한다면, 아주 빠른 속도로 batch 단위의 문장들로 Encoding 할 수 있습니다.


```python
sentences = []

with open('./sample_corpus.txt', 'r') as f:
    for line in f:
        sentences.append(line)
```


```python
encodings = bert_wordpiece_tokenizer.encode_batch(sentences)

for i in range(len(sentences)):
    print(f'\nSentence:', sentences[i])
    print(f'Tokens:', encodings[i].tokens)
    print(f'Ids:', encodings[i].ids)
```

    
    Sentence: Joe waited for the train.
    
    Tokens: ['j', '##o', '##e', 'w', '##a', '##i', '##t', '##e', '##d', 'f', '##o', '##r', 't', '##h', '##e', 't', '##r', '##a', '##i', '##n', '.']
    Ids: [13, 29, 27, 23, 25, 28, 26, 27, 34, 10, 29, 31, 21, 33, 27, 21, 31, 25, 28, 30, 5]
    
    Sentence: The train was late.
    
    Tokens: ['t', '##h', '##e', 't', '##r', '##a', '##i', '##n', 'w', '##a', '##s', 'l', '##a', '##t', '##e', '.']
    Ids: [21, 33, 27, 21, 31, 25, 28, 30, 23, 25, 36, 15, 25, 26, 27, 5]
    
    Sentence: Mary and Samantha took the bus.
    
    Tokens: ['m', '##a', '##r', '##y', 'a', '##n', '##d', 's', '##a', '##m', '##a', '##n', '##t', '##h', '##a', 't', '##o', '##o', '##k', 't', '##h', '##e', 'b', '##u', '##s', '.']
    Ids: [16, 25, 31, 32, 6, 30, 34, 20, 25, 37, 25, 30, 26, 33, 25, 21, 29, 29, 38, 21, 33, 27, 7, 35, 36, 5]
    
    Sentence: I looked for Mary and Samantha at the bus station
    
    Tokens: ['i', 'l', '##o', '##o', '##k', '##e', '##d', 'f', '##o', '##r', 'm', '##a', '##r', '##y', 'a', '##n', '##d', 's', '##a', '##m', '##a', '##n', '##t', '##h', '##a', 'a', '##t', 't', '##h', '##e', 'b', '##u', '##s', 's', '##t', '##a', '##t', '##i', '##o', '##n']
    Ids: [12, 15, 29, 29, 38, 27, 34, 10, 29, 31, 16, 25, 31, 32, 6, 30, 34, 20, 25, 37, 25, 30, 26, 33, 25, 6, 26, 21, 33, 27, 7, 35, 36, 20, 26, 25, 26, 28, 29, 30]


이에 따른 결과를 통해서 좀 더 파라미터를 자세히 살펴보겠습니다. 당연히 `vocab_size`를 늘리면 더 많은 subwords가 vocab으로 학습됩니다.


```python
bert_wordpiece_tokenizer.train(
    files = './sample_corpus.txt',
    vocab_size = 100, #from 30 to 100
    min_frequency = 1,
    limit_alphabet = 1000,
    initial_alphabet = [],
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    show_progress = True,
    wordpieces_prefix = "##",
)
```


```python
encodings = bert_wordpiece_tokenizer.encode_batch(sentences)

for i in range(len(sentences)):
    print(f'\nSentence:', sentences[i])
    print(f'Tokens:', encodings[i].tokens)
    print(f'Ids:', encodings[i].ids)
```

    
    Sentence: Joe waited for the train.
    
    Tokens: ['joe', 'waited', 'for', 'the', 'train', '.']
    Ids: [77, 82, 58, 40, 61, 5]
    
    Sentence: The train was late.
    
    Tokens: ['the', 'train', 'was', 'late', '.']
    Ids: [40, 61, 81, 78, 5]
    
    Sentence: Mary and Samantha took the bus.
    
    Tokens: ['mary', 'and', 'samantha', 'took', 'the', 'bus', '.']
    Ids: [59, 56, 64, 70, 40, 57, 5]
    
    Sentence: I looked for Mary and Samantha at the bus station
    
    Tokens: ['i', 'looked', 'for', 'mary', 'and', 'samantha', 'at', 'the', 'bus', 'station']
    Ids: [12, 79, 58, 59, 56, 64, 65, 40, 57, 80]


### Model Save and Load 

`save_model`을 이용하여 {directory}/{name}-vocab.txt 파일로 vocab 을 저장합니다.
tokenizer 종류에 따라서 저장되는 결과들은 달라집니다.

```python
# save tokenizer
bert_wordpiece_tokenizer.save_model(
    directory='./tokenizer/',
    name = 'example_bertwordpiece'
)
```

    ['./tokenizer/example_bertwordpiece-vocab.txt']

tokenizer를 다시 불러오기 위해서는 저장된 vocab파일만 가져오기만 하면 됩니다.

```python
# load tokenizer
bert_wordpiece_tokenizer = BertWordPieceTokenizer(
    vocab_file = './tokenizer/example_bertwordpiece-vocab.txt'
)
```

### Special Tokens

encode를 수행할 때, `add_special_tokens`에 따라서 speical_tokens를 출력할지 결정할 수 있습니다.

```python
bert_wordpiece_tokenizer.encode('I looked for Mary and Samantha at the bus station').tokens
```




    ['[CLS]',
     'i',
     'looked',
     'for',
     'mary',
     'and',
     'samantha',
     'at',
     'the',
     'bus',
     'station',
     '[SEP]']




```python
bert_wordpiece_tokenizer.encode('I looked for Mary and Samantha at the bus station', add_special_tokens=False).tokens
```




    ['i',
     'looked',
     'for',
     'mary',
     'and',
     'samantha',
     'at',
     'the',
     'bus',
     'station']



또한 BERT는 두 문장을 `[SEP]`으로 구분하기 때문에, Pair기능을 제공합니다.


```python
bert_wordpiece_tokenizer.encode(
    sequence = 'Joe waited for the train.',
    pair = 'The train was late.'
).tokens
```




    ['[CLS]',
     'joe',
     'waited',
     'for',
     'the',
     'train',
     '.',
     '[SEP]',
     'the',
     'train',
     'was',
     'late',
     '.',
     '[SEP]']



### Add Tokens

학습된 Tokenizer에서 Token을 추가하기 위해서는 `add_tokens`를 이용해서 직접 추가할 수 있습니다.

```python
bert_wordpiece_tokenizer.add_tokens(['airplane'])

bert_wordpiece_tokenizer.save_model(
    directory='./tokenizer/',
    name = 'example_bertwordpiece2'
)

bert_wordpiece_tokenizer = BertWordPieceTokenizer(
    vocab_file = './tokenizer/example_bertwordpiece2-vocab.txt'
)
```


```python
bert_wordpiece_tokenizer.encode('Joe waited for the airplane.').tokens
```




    ['[CLS]', 'joe', 'waited', 'for', 'the', '[UNK]', '.', '[SEP]']

하지만 위의 예제와 같이 저장이 제대로 되지 않습니다.

따라서 아래처럼 vocab파일에 직접 추가해주어야 합니다.


```python
with open('./tokenizer/example_bertwordpiece2-vocab.txt', 'a') as f:
        f.write('airplane')
```


```python
bert_wordpiece_tokenizer = BertWordPieceTokenizer(
    vocab_file = './tokenizer/example_bertwordpiece2-vocab.txt'
)
```


```python
bert_wordpiece_tokenizer.encode('Joe waited for the airplane.').tokens
```




    ['[CLS]', 'joe', 'waited', 'for', 'the', 'airplane', '.', '[SEP]']

### 학습된 Tokenizer를 transformers에서 이용하기

```python
from transformers import BertTokenizer

transforms_bert_tokenizer = BertTokenizer(
    vocab_file = './tokenizer/example_bertwordpiece2-vocab.txt'
)

sentence = 'Mary waited for the airplane.'

print(f'Transformers: {transforms_bert_tokenizer.tokenize(sentence)}')
```

    Transformers: ['mary', 'waited', 'for', 'the', 'airplane', '.']


## SentencePieceBPE Tokenizer

SentencePiece는 공백 뒤에 등장하는 단어 앞에 `_`를 붙여 실제 공백과 subwords의 경계를 구분합니다.


```python
from tokenizers import SentencePieceBPETokenizer

sentencepiece_tokenizer = SentencePieceBPETokenizer(
    add_prefix_space=True
)
```

`add_prefix_space`가 True이면, 문장의 맨 앞 단어에도 공백을 부여합니다.
False일 경우, 공백없이 시작하는 단어에는 `_`를 부여하지 않습니다.


```python
sentencepiece_tokenizer.train(
    files = './sample_corpus.txt',
    vocab_size = 50,
    min_frequency = 1,
    special_tokens = ['<unk>'],
)
```


```python
vocab = sentencepiece_tokenizer.get_vocab()
sorted(vocab, key=lambda x: vocab[x])
```




    ['<unk>',
     '.',
     'I',
     'J',
     'M',
     'S',
     'T',
     'a',
     'b',
     'd',
     'e',
     'f',
     'h',
     'i',
     'k',
     'l',
     'm',
     'n',
     'o',
     'r',
     's',
     't',
     'u',
     'w',
     'y',
     '▁',
     '▁t',
     'an',
     'he',
     'ai',
     'at',
     '▁the',
     'Ma',
     'Sa',
     'bu',
     'ed',
     'fo',
     'ha',
     'man',
     'ok',
     'ook',
     'ry',
     'rai',
     'tha',
     '▁l',
     '▁w',
     '▁an',
     '▁Ma',
     '▁Sa',
     '▁bu']




```python
sentencepiece_tokenizer.encode('Joe waited for the train.').tokens
```




    ['▁',
     'J',
     'o',
     'e',
     '▁w',
     'ai',
     't',
     'ed',
     '▁',
     'fo',
     'r',
     '▁the',
     '▁t',
     'rai',
     'n',
     '.']



### Model Save and Load


```python
sentencepiece_tokenizer.save_model('./vocab', 'example_sentencepiece')
```




    ['./tokenizer/example_sentencepiece-vocab.json',
     './tokenizer/example_sentencepiece-merges.txt']



BPE기반의 Tokenizer들은 vocab.json, merges.txt 두 개의 파일을 저장합니다.
따라서 학습된 Tokenizer들을 이용하기 위해서 두 개의 파일을 모두 로드해야 합니다.


```python
sentencepiece_tokenizer = SentencePieceBPETokenizer(
    vocab_file = './tokenizer/example_sentencepiece-vocab.json',
    merges_file = './tokenizer/example_sentencepiece-merges.txt'
)
```


```python
sentencepiece_tokenizer.encode('Joe waited for the airplane.').tokens
```




    ['▁',
     'J',
     'o',
     'e',
     '▁w',
     'ai',
     't',
     'ed',
     '▁',
     'fo',
     'r',
     '▁the',
     '▁',
     'ai',
     'r',
     '<unk>',
     'l',
     'an',
     'e',
     '.']

## ByteLevelTokenizer

Byte-Level BPE는 글자가 아닌 byte 기준으로 BPE를 적용하기 때문에, 1byte로 표현되는 글자(알파벳, 숫자, 기호)만 형태가 보존됩니다.


```python
from tokenizers import ByteLevelBPETokenizer

bytebpe_tokenizer = ByteLevelBPETokenizer()
```


```python
bytebpe_tokenizer.train(
    files = './sample_corpus.txt',
    vocab_size = 100,
    min_frequency = 1,
)
```


```python
vocab = bytebpe_tokenizer.get_vocab()
print(sorted(vocab, key=lambda x: vocab[x]))
```

    ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '¡', '¢', '£', '¤', '¥', '¦', '§', '¨', '©', 'ª', '«', '¬', '®', '¯', '°', '±', '²', '³', '´', 'µ', '¶', '·', '¸', '¹', 'º', '»', '¼', '½', '¾', '¿', 'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Æ', 'Ç', 'È', 'É', 'Ê', 'Ë', 'Ì', 'Í', 'Î', 'Ï', 'Ð', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', '×', 'Ø', 'Ù', 'Ú', 'Û', 'Ü', 'Ý', 'Þ', 'ß', 'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ð', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', '÷', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ', 'Ā', 'ā', 'Ă', 'ă', 'Ą', 'ą', 'Ć', 'ć', 'Ĉ', 'ĉ', 'Ċ', 'ċ', 'Č', 'č', 'Ď', 'ď', 'Đ', 'đ', 'Ē', 'ē', 'Ĕ', 'ĕ', 'Ė', 'ė', 'Ę', 'ę', 'Ě', 'ě', 'Ĝ', 'ĝ', 'Ğ', 'ğ', 'Ġ', 'ġ', 'Ģ', 'ģ', 'Ĥ', 'ĥ', 'Ħ', 'ħ', 'Ĩ', 'ĩ', 'Ī', 'ī', 'Ĭ', 'ĭ', 'Į', 'į', 'İ', 'ı', 'Ĳ', 'ĳ', 'Ĵ', 'ĵ', 'Ķ', 'ķ', 'ĸ', 'Ĺ', 'ĺ', 'Ļ', 'ļ', 'Ľ', 'ľ', 'Ŀ', 'ŀ', 'Ł', 'ł', 'Ń']


띄어쓰기로 시작하는 단어 앞에 `Ġ`를 prefix로 부착합니다.


```python
bytebpe_tokenizer.encode('Joe waited for the train.').tokens
```




    ['J',
     'o',
     'e',
     'Ġ', - prefix
     'w',
     'a',
     'i',
     't',
     'e',
     'd',
     'Ġ', - prefix
     'f',
     'o',
     'r',
     'Ġ', - prefix
     't',
     'h',
     'e',
     'Ġ', - prefix
     't',
     'r',
     'a',
     'i',
     'n',
     '.']

## CharBPETokenizer

Character-level BPE는 단어 수준에서 BPE를 이용하여 subwords를 학습하며, 단어에 suffix로 `</w>`를 부착하여 공백을 표현합니다.


```python
from tokenizers import CharBPETokenizer

charbpe_tokenizer = CharBPETokenizer(
    suffix = '</w>',
    split_on_whitespace_only=True
)
```

CharBPETokenizer는 기본적으로 공백과 구두점을 이용하여 텍스트를 분리합니다. 
그렇기 때문에 문장이 `.`으로 끝나는 경우를 공백으로 나타내지 않기 위해 `split_on_white_space_only`옵션을 True로 설정해줍니다.

```python
charbpe_tokenizer.train(
    files = './sample_corpus.txt',
    vocab_size = 50,
    min_frequency = 1,
)
```

```python
charbpe_tokenizer.encode('Joe waited for the train.').tokens
```
    ['J',
     'o',
     'e</w>', - suffix
     'w',
     'ai',
     't',
     'ed</w>', - suffix
     'fo',
     'r</w>', - suffix
     'the</w>', - suffix
     't',
     'rai',
     'n',
     '.</w>'- suffix
     ] 

`split_on_white_space_only`옵션을 False로 설정했을 경우, 아래의 예문 안에 `.`앞에서 공백으로 표현합니다.


```python
charbpe_tokenizer = CharBPETokenizer(
    suffix = '</w>',
)

charbpe_tokenizer.train(
    files = './sample_corpus.txt',
    vocab_size = 50,
    min_frequency = 1,
)

charbpe_tokenizer.encode('Joe waited for the train.').tokens
```
    ['J',
     'o',
     'e</w>', - suffix
     'w',
     'ai',
     't',
     'ed</w>', - suffix
     'fo',
     'r</w>', - suffix
     'the</w>', - suffix
     't',
     'rai',
     'n</w>', - suffix
     '.</w>' - suffix
     ]



## Tokenizer Result

위에서 수행한 Tokenizer들을 하나의 문장을 통해서 그에 따른 결과들을 살펴보겠습니다.

```python
sentence = 'Joe waited for the airplane.'

tokenizers = [bert_wordpiece_tokenizer, 
              sentencepiece_tokenizer, 
              charbpe_tokenizer, 
              bytebpe_tokenizer]

for tokenizer in tokenizers:
    encode_single = tokenizer.encode(sentence)
    print(f'\n{tokenizer.__class__.__name__}')
    print(f'tokens = {encode_single.tokens}')
```

    
    BertWordPieceTokenizer
    tokens = ['[CLS]', 'joe', 'waited', 'for', 'the', 'airplane', '.', '[SEP]']
    
    SentencePieceBPETokenizer
    tokens = ['▁', 'J', 'o', 'e', '▁w', 'ai', 't', 'ed', '▁', 'fo', 'r', '▁the', '▁', 'ai', 'r', '<unk>', 'l', 'an', 'e', '.']
    
    CharBPETokenizer
    tokens = ['J', 'o', 'e</w>', 'w', 'ai', 't', 'ed</w>', 'fo', 'r</w>', 'the</w>', 'ai', 'r', 'l', 'an', 'e</w>', '.</w>']
    
    ByteLevelBPETokenizer
    tokens = ['J', 'o', 'e', 'Ġ', 'w', 'a', 'i', 't', 'e', 'd', 'Ġ', 'f', 'o', 'r', 'Ġ', 't', 'h', 'e', 'Ġ', 'a', 'i', 'r', 'p', 'l', 'a', 'n', 'e', '.']

