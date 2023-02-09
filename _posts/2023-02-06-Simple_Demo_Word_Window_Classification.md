---
title: "Simple Word Window Classifier"
last_modified_at: 2023-01-16
categories:
  - NLP
tags:
  - NLP
  - Pytorch
---

This demo is mostly a rewrite version of the tutorial [Stanford CS 224N](https://colab.research.google.com/drive/13HGy3-uIIy1KD_WFhG4nVrxJC-3nUUkP?usp=sharing), with minor edition and some analysis.

# Problem description
build a simple classifier input fixed number of words and output whether the center word is a LOCATION.


```python
# Our raw data, which consists of sentences
corpus = [
          "We always come to Paris",
          "The professor is from Australia",
          "I live in Stanford",
          "He comes from Taiwan",
          "The capital of Turkey is Ankara"
         ]
```

# preprocessing
* special characters
* tokenization
* lowercasing


```python
import re
s = "We! always come to Paris.12" # only keep letters
re.sub(r'[^A-Za-z ]+', '', s)
```




    'We always come to Paris'




```python
# simple lowercase all and split(by space) into words
def preprocess_sentence(sentence):
  return re.sub(r'[^A-Za-z ]+', '',sentence).lower().split()

train_sentences = [preprocess_sentence(sent) for sent in corpus]
train_sentences
```




    [['we', 'always', 'come', 'to', 'paris'],
     ['the', 'professor', 'is', 'from', 'australia'],
     ['i', 'live', 'in', 'stanford'],
     ['he', 'comes', 'from', 'taiwan'],
     ['the', 'capital', 'of', 'turkey', 'is', 'ankara']]



## Generating labels for training data:
* if the word is a LOCATION, label 1
* else, label 0.


```python
# Set of locations that appear in our corpus
locations = set(["australia", "ankara", "paris", "stanford", "taiwan", "turkey"])

# Our train labels
train_labels = [[1 if word in locations else 0 for word in sent] for sent in train_sentences]
train_labels
```




    [[0, 0, 0, 0, 1],
     [0, 0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 1, 0, 1]]



## build a dictionary


```python
vocabulary = set(w for s in train_sentences for w in s)
vocabulary.add("<unk>") # add the unknown token
vocabulary.add("<pad>") # add the padding for window
len(vocabulary)
```




    23




```python
# notice that the total window size is 2*window_size+1
def pad_window(sentence, window_size, pad_token = "<pad>"):
  window = [pad_token] * window_size
  return window + sentence + window

pad_window(train_sentences[1], window_size=2)
```




    ['<pad>',
     '<pad>',
     'the',
     'professor',
     'is',
     'from',
     'australia',
     '<pad>',
     '<pad>']




```python
idx_to_word = sorted(list(vocabulary)) 

word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}

def convert_token_to_idx(sentence, word_to_idx):
  return [word_to_idx.get(token, word_to_idx["<unk>"]) for token in sentence]

print(f"From token list to indices: {convert_token_to_idx(train_sentences[0], word_to_idx)}")
```

    From token list to indices: [22, 2, 6, 20, 15]


## Create an embedding table

with `nn.Embedding(num_words, embedding_dimension)`


```python
import torch
import torch.nn as nn
```


```python
embedding_dim = 5 # embedding dimension is 5
embeds = nn.Embedding(len(vocabulary), embedding_dim) # create an embedding table

list(embeds.named_parameters()) # current embeddings for each word
```




    [('weight', Parameter containing:
      tensor([[ 0.4098, -2.4334,  0.3162,  0.2969,  0.1592],
              [-1.2451, -2.5765, -1.6796,  0.7516, -0.7779],
              [-1.8351,  0.0980, -0.0330,  0.8593, -1.9052],
              [ 1.2093, -0.3367, -0.8238, -0.7045, -1.0983],
              [ 0.6358, -0.2363,  1.3068, -1.1299,  0.5609],
              [ 0.3433,  0.2411, -0.7985,  1.3960,  0.9617],
              [-0.3141, -0.1031,  0.1062,  1.4690,  0.8606],
              [-0.2018, -0.7244,  0.3078, -0.0094,  1.0512],
              [ 2.1535,  1.1693,  0.0591,  0.3641,  0.2246],
              [-0.7594, -0.4067, -0.9263,  1.6237, -0.7148],
              [-0.4435,  0.3144,  0.1890,  1.3535,  0.0840],
              [-0.2986,  2.7067,  0.2760,  0.3518, -1.2447],
              [ 0.2308,  0.5228, -1.0962, -0.2783,  0.3644],
              [ 0.3816, -0.8633, -1.0878, -0.7087,  0.1341],
              [-1.9324,  0.9979, -4.0401, -0.3142, -0.2032],
              [ 0.1987,  0.3541,  0.0593, -1.7592,  0.5147],
              [ 0.3165, -0.1248,  0.9690, -1.4124,  0.1278],
              [ 0.0466,  0.2293,  0.1853,  0.9252, -1.2548],
              [ 0.6421, -0.0980, -0.0566, -2.0558, -0.0834],
              [ 0.1235, -0.1304, -1.2258,  1.0515, -0.9185],
              [-0.1147,  0.2625, -0.7694,  0.3040, -0.3957],
              [-1.0098, -2.3781, -0.1872, -0.8269,  0.4008],
              [-1.0670,  0.5430, -0.9485,  0.1729, -1.1032]], requires_grad=True))]




```python
# Get embeddings for words (if we want to do something later...)
indices = torch.tensor([word_to_idx[v] for v in ["paris", "ankara"]], dtype=torch.long)
embeddings = embeds(indices)
embeddings
```




    tensor([[ 0.1987,  0.3541,  0.0593, -1.7592,  0.5147],
            [ 1.2093, -0.3367, -0.8238, -0.7045, -1.0983]],
           grad_fn=<EmbeddingBackward0>)



## Batching Sentences

`DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)`

in `collate_fn`, we can put a custom function.


```python
from torch.utils.data import DataLoader
from functools import partial
```


```python
def custom_collate_fn(batch, window_size, word_to_idx):
  x, y = zip(*batch)

  def pad_window(sentence, window_size, pad_token="<pad>"):
    window = [pad_token]*window_size
    return window + sentence +window

  x = [pad_window(s, window_size=window_size) for s in x]

  def convert_token_to_idx(sentence, word_to_idx):
    return [word_to_idx.get(token, word_to_idx["<unk>"]) for token in sentence] 
    # use get here to have a default value for words not in dictionary
  
  x = [convert_token_to_idx(s, word_to_idx) for s in x]
  pad_token_idx = word_to_idx["<pad>"]

  # pad all sentences to equal length
  x = [torch.LongTensor(x_i) for x_i in x]
  x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=pad_token_idx)
  
  lengths = [len(label) for label in y]
  lengths = torch.LongTensor(lengths)

  y = [torch.LongTensor(y_i) for y_i in y]
  y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=pad_token_idx)

  return x_padded, y_padded, lengths

```


```python
# parameters
data = list(zip(train_sentences, train_labels))
batch_size = 2
shuffle = True
window_size = 2
collate_fn = partial(custom_collate_fn, window_size=window_size, word_to_idx=word_to_idx)

# instantiate
loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

counter = 0
for batched_x, batched_y, batched_lengths in loader:
  print(f"Iteration {counter}")
  print("Batched Input:")
  print(batched_x)
  print("Batched Labels:")
  print(batched_y)
  print("Batched Lengths:")
  print(batched_lengths)
  print("")
  counter += 1
```

    Iteration 0
    Batched Input:
    tensor([[ 0,  0, 22,  2,  6, 20, 15,  0,  0,  0],
            [ 0,  0, 19,  5, 14, 21, 12,  3,  0,  0]])
    Batched Labels:
    tensor([[0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1]])
    Batched Lengths:
    tensor([5, 6])
    
    Iteration 1
    Batched Input:
    tensor([[ 0,  0, 19, 16, 12,  8,  4,  0,  0],
            [ 0,  0,  9,  7,  8, 18,  0,  0,  0]])
    Batched Labels:
    tensor([[0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0]])
    Batched Lengths:
    tensor([5, 4])
    
    Iteration 2
    Batched Input:
    tensor([[ 0,  0, 10, 13, 11, 17,  0,  0]])
    Batched Labels:
    tensor([[0, 0, 0, 1]])
    Batched Lengths:
    tensor([4])
    



```python
# create windows using unfold function
chunk = batched_x.unfold(1, window_size*2+1, 1)
print(chunk)
```

    tensor([[[ 0,  0, 10, 13, 11],
             [ 0, 10, 13, 11, 17],
             [10, 13, 11, 17,  0],
             [13, 11, 17,  0,  0]]])


# Model


```python
class WordWindowClassifier(nn.Module):
  def __init__(self, param, vocab_size, pad_idx=0):
    super(WordWindowClassifier, self).__init__()

    self.window_size = param["window_size"]
    self.embed_dim = param["embed_dim"]
    self.hidden_dim = param["hidden_dim"]
    self.freeze_embeddings = param["freeze_embeddings"]

    # embedding layer
    self.embeds = nn.Embedding(vocab_size, self.embed_dim, padding_idx=pad_idx)
    # if freeze_embeddings, set require grad to false
    if self.freeze_embeddings:
      self.embed_layer.weight.requires_grad = False 

    """ Hidden Layer
    """
    full_window_size = 2*window_size+1
    self.hidden_layer = nn.Sequential(
        nn.Linear(full_window_size * self.embed_dim, self.hidden_dim),
        nn.Tanh()
    )  

    """ Output Layer
    """
    self.output_layer = nn.Linear(self.hidden_dim, 1)

    self.prob = nn.Sigmoid()

  def forward(self, inputs):
    B, L = inputs.size()

    token_windows = inputs.unfold(1, 2*self.window_size+1, 1)
    _,adjusted_length,_ = token_windows.size()

    assert token_windows.size() == (B, adjusted_length, 2*self.window_size+1)

    # embedding layer
    embedded_windows = self.embeds(token_windows)

    # reshape to combine dim of windows and embeddings
    embedded_windows = embedded_windows.view(B, adjusted_length, -1)

    layer_1 = self.hidden_layer(embedded_windows)

    output = self.output_layer(layer_1)

    output = self.prob(output)
    output = output.view(B, -1)
    
    return output
```

# Training


```python
data = list(zip(train_sentences, train_labels))
batch_size = 2
shuffle = True
window_size = 2
collate_fn = partial(custom_collate_fn, window_size=window_size, word_to_idx=word_to_idx)

loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
                    )

model_param = {
    "batch_size": 4,
    "window_size": 2,
    "embed_dim": 25,
    "hidden_dim": 25,
    "freeze_embeddings": False
}

vocab_size = len(word_to_idx)
model = WordWindowClassifier(model_param, vocab_size)

# optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# loss function
def loss_function(batch_outputs, batch_labels, batch_lengths):
  bceloss = nn.BCELoss()
  loss = bceloss(batch_outputs, batch_labels.float())

  loss = loss/batch_labels.sum().float()

  return loss
```


```python
def train_epoch(loss_function, optimizer, model, loader):

  total_loss = 0
  for batch_inputs, batch_labels, batch_lengths in loader:
    # clear gradients
    optimizer.zero_grad()
    # forward pass
    outputs = model.forward(batch_inputs)
    # compute loss
    loss = loss_function(outputs, batch_labels, batch_lengths)
    # gradients
    loss.backward()
    # update params
    optimizer.step()
    total_loss += loss.item()

  return total_loss

def train(loss_function, optimizer, model, loader, num_epochs=10000):
  for epoch in range(num_epochs):
    epoch_loss = train_epoch(loss_function, optimizer, model, loader)
    if epoch%100 == 0:
      print(epoch_loss)
```


```python
num_epochs = 1000
train(loss_function, optimizer, model, loader, num_epochs=num_epochs)
```

    0.002883558685425669
    0.0028394981054589152
    0.002706396917346865
    0.0038161433476489037
    0.0027932398952543736
    0.0022353382664732635
    0.003413549275137484
    0.002417302515823394
    0.0021423909347504377
    0.002302502456586808


# Make predictions


```python
test_corpus = ["She comes from Paris",
               "She comes from China"]
test_sentences = [preprocess_sentence(sent) for sent in test_corpus]
test_labels = [[0, 0, 0, 1],[0, 0, 0, 1]]

test_data = list(zip(test_sentences, test_labels))
batch_size = 1
shuffle = False
window_size = 2
collate_fn = partial(custom_collate_fn, window_size=2, word_to_idx=word_to_idx)
test_loader = torch.utils.data.DataLoader(test_data, 
                                           batch_size=1, 
                                           shuffle=False, 
                                           collate_fn=collate_fn)
```


```python
for test_instances, labels, _ in test_loader:
  outputs = model.forward(test_instances)
  print(labels)
  print(outputs)
```

    tensor([[0, 0, 0, 1]])
    tensor([[8.4251e-03, 1.5757e-04, 1.6452e-04, 9.9932e-01]],
           grad_fn=<ViewBackward0>)
    tensor([[0, 0, 0, 1]])
    tensor([[8.4251e-03, 7.1489e-04, 3.7368e-04, 9.9879e-01]],
           grad_fn=<ViewBackward0>)


# Result analysis


```python
predict_probs = []
for test_instances, labels, _ in test_loader:
  outputs = model.forward(test_instances)
  predict_probs.append(outputs.detach().numpy())
  print(labels)
  print(outputs)
```

    tensor([[0, 0, 0, 1]])
    tensor([[8.4251e-03, 1.5757e-04, 1.6452e-04, 9.9932e-01]],
           grad_fn=<ViewBackward0>)
    tensor([[0, 0, 0, 1]])
    tensor([[8.4251e-03, 7.1489e-04, 3.7368e-04, 9.9879e-01]],
           grad_fn=<ViewBackward0>)



```python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML

def colorize(words, color_array):
    cmap=matplotlib.cm.RdYlGn
    template = '<span class="barcode"; style="color: white; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string

# # or simply save in an html file and open in browser
# with open('colorize.html', 'w') as f:
#     f.write(s)
```


```python
for sentence, prediction in zip(test_sentences, predict_probs):
  s = colorize(sentence, prediction[0])
  display(HTML(s))
```


<span class="barcode"; style="color: white; background-color: #a90426">&nbspshe&nbsp</span><span class="barcode"; style="color: white; background-color: #a50026">&nbspcomes&nbsp</span><span class="barcode"; style="color: white; background-color: #a50026">&nbspfrom&nbsp</span><span class="barcode"; style="color: white; background-color: #006837">&nbspparis&nbsp</span>



<span class="barcode"; style="color: white; background-color: #a90426">&nbspshe&nbsp</span><span class="barcode"; style="color: white; background-color: #a50026">&nbspcomes&nbsp</span><span class="barcode"; style="color: white; background-color: #a50026">&nbspfrom&nbsp</span><span class="barcode"; style="color: white; background-color: #006837">&nbspchina&nbsp</span>


From those 2 simple test sentences, we could see the toy classifier does well on both the LOCATION words in dictionary("Paris") and not in the dictionary("China").


```python

```
