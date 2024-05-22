# text-generation-GRU
## Описание задачи
В данной работе перед нами стоит задача в построении языковой модели с использованием рекуррентных нейронных сетей с элементами GRU (Gated Recurrent Unit).
Для этого нам потребуется осуществить подготовку данных, а затем построить и обучить рекуррентную нейронную сеть с использованием GRU для создания языковой модели. Далее мы будем исследовать различные аспекты генерации текста с использованием этой модели, а именно вариации температуры и длины текста, а также дадим оценку перплексии для анализа качества генерации.
## Листинг кода с пояснениями
```
import tensorflow as tf
import numpy as np
import os
import time
import urllib.request
import re
tf.__version__
```
### Предобработка данных:
1) Загрузка книги
```
url = "https://www.gutenberg.org/cache/epub/64317/pg64317.txt"
file = urllib.request.urlopen(url)
text = [line.decode('utf-8') for line in file]
text = ''.join(text)
text = re.sub(' +',' ',text)
text = re.sub(r'[^A-Za-z.,!\r ]+', '', text)
text = text[1140:]
text[:500]
```

3) Токенизация и кодирование<br>
Извлечение словаря токенов символов из текста
```
vocab = sorted(set(text))
",".join(vocab)
 ```
4) Кодировка символов и функций отображения<br>
Функции отображения char2idx и idx2char отображают символы в индексы и обратно
```
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
[f"{char} = {i}" for char,i in zip(char2idx, range(20))]
```
4) Построение обучающих наборов<br>
Переменная example_per_epoch — это количество выборок или фрагментов текста, которые мы будем передавать модели. char_dataset — это преобразование кодировок text_as_int в тензоры.
```
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
[idx2char[i.numpy()] for i in char_dataset.take(5)]
```
5) Разбиение текста на обучающие последовательности
```
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
[repr(''.join(idx2char[item.numpy()])) for item in sequences.take(6)]
 
```
6) Создание входной и таргетовой последовательности с помощью простой функции карты<br>
Поскольку мы обучаем сеть последовательностям, создадим входную последовательность, а затем таргетовую или целевую. При использовании RNN целевой последовательностью будет входная последовательность, смещенная на один символ
```
@tf.autograph.experimental.do_not_convert
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
dataset = sequences.map(split_input_target)
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
 ```

7) Пример входных и ожидаемых выходных данных, к которым будем обучать сеть
```
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
 ```
### Построение и обучение модели
Гиперпараметры для модели
```
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
rnn_units_2 = 512
```
Создание модели
```
model = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, embedding_dim,
                            batch_input_shape=[BATCH_SIZE, None]),
  tf.keras.layers.GRU(rnn_units,
                      return_sequences=True,
                      stateful=True,
                      recurrent_initializer='glorot_uniform'),
  tf.keras.layers.GRU(rnn_units_2,
                      return_sequences=True,
                      stateful=True,
                      recurrent_initializer='glorot_uniform'),
  tf.keras.layers.Dense(vocab_size)
])
model.summary()
```
В этой модели используются 2 уровня RNN типа GRU или вентилируемого рекуррентного блока. Уровни GRU проще, чем LSTM, и не требуют ввода состояния. <br>

Определение функции потерь
```
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
model.compile(optimizer='adam', loss=loss)
```
Сохранение копии модели для дальнейшего использования
```
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
```
Обучение модели
```
epochs = 5
history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
```
Важно, что обучение RNN дорого обходится и может занять много времени. Мы тренируем здесь только 5 эпох в демонстрационных целях. Чтобы получить хорошо настроенную модель, этот пример лучше всего запустить с 50 эпохами<br>

 Определение функции для запроса модели и генерации текста
```
def generate_text(model, start_string, temp, gen_chars):
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  model.reset_states()
  for i in range(gen_chars):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)
    predictions = predictions / temp
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    input_eval = tf.expand_dims([predicted_id], 0)
    text_generated.append(idx2char[predicted_id])
  return (start_string + ''.join(text_generated))
```
Перестройка модели, используя только 1 вход или батч
```
model = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, embedding_dim,
                            batch_input_shape=[1, None]),
  tf.keras.layers.GRU(rnn_units,
                      return_sequences=True,
                      stateful=True,
                      recurrent_initializer='glorot_uniform'),
  tf.keras.layers.GRU(rnn_units_2,
                      return_sequences=True,
                      stateful=True,
                      recurrent_initializer='glorot_uniform'),
  tf.keras.layers.Dense(vocab_size)])
model.summary()
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
```
 

### Генерация текстов разных температур<br> 
Температура используется для определения предсказуемости текста. Более низкая температура (0,25) создает интеллектуальный текст. В то время как более высокая температура (2.0) генерирует более уникальный текст. Более высокие температуры могут привести к бессмысленному тексту.

```
text_025 = generate_text(model, u"He said ", .25, 200)
 
text_05 = generate_text(model, u"She thinks ", 0.5, 200)
 
text_08 = generate_text(model, u"In my younger ", 0.8, 200)
 
text_1 = generate_text(model, u"I never ", 1.0, 200)
 
text_2 = generate_text(model, u"I never ", 2.0, 200)
``` 

### Перплексия и энтропия<br>
Лучше та модель, которая лучше предсказывает детали тестовой коллекции (меньше перплексия)
```
from nltk import ngrams
from collections import Counter
import math
corpus = text_025
n = 2
ngram_counts = Counter(ngrams(corpus.split(), n))
total_count = sum(ngram_counts.values())
probabilities = {ngram: count/total_count for ngram, count in ngram_counts.items()}

def calculate_perplexity(probabilities):
    entropy = 0.0
    for prob in probabilities:
        entropy += math.log2(prob)
    perplexity = 2 ** (-entropy / len(probabilities))
    return perplexity
probs = []
print("Вероятности биграмм:")
for ngram, prob in probabilities.items():
    print(' '.join(ngram), "->", prob)
 
probabilities = probs
perplexity_1 = calculate_perplexity(probabilities)
print("Perplexity:", perplexity_1)

```
### Генерация текстов разной длины
```
text_1 = generate_text(model, u"He said ", 0.5, 50)
 
text_2 = generate_text(model, u"She thinks ", 0.5, 100)
 
text_3 = generate_text(model, u"In my younger ", 0.5, 150)
 
text_4 = generate_text(model, u"I never ", 0.5, 250)
 
text_5 = generate_text(model, u"I never ", 0.5, 500)
 
corpus = text_1
n = 2
ngram_counts = Counter(ngrams(corpus.split(), n))
total_count = sum(ngram_counts.values())
probabilities = {ngram: count/total_count for ngram, count in ngram_counts.items()}
probs = []
print("Вероятности биграмм:")
for ngram, prob in probabilities.items():
    print(' '.join(ngram), "->", prob)
 
probabilities = probs
perplexity_1 = calculate_perplexity(probabilities)
print("Perplexity:", perplexity_1)

```

### Определение потерь
```
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label = 'Обучение')
plt.xlabel('Эпоха обучения')
plt.ylabel('Потери')
plt.legend()
plt.show()
```
 График перплексии и температуры:<br>
![image](https://github.com/cranberriess/text-generation-GRU/assets/105839329/93f40e65-98ca-4c38-be22-1233887b65c4)<br>
 График перплексии и длины:<br>
![image](https://github.com/cranberriess/text-generation-GRU/assets/105839329/65690867-b67a-449a-8d22-837aeb13f63a)<br>
График потерь:<br>
 ![image](https://github.com/cranberriess/text-generation-GRU/assets/105839329/69b77088-ac59-4daf-ba4f-9a6884fafb4c)<br>

## Выводы на основе графиков 
Проанализировав 1 график (см. «генерация текстов разных температур») можно сделать вывод, что 
•	тексты при температурах 0.25 и 1 имеют схожие значения перплексии 38 и 39 соответственно. Эти показатели относительно низкие, что обычно свидетельствует о лучшей читаемости сгенерированных текстов.
•	тексты при температурах 0.5 и 0.8 имеют высокие значения перплексии 44 и 44.2 соответственно. Это указывает на более низкое качество генерации.
•	текст при самой высокой температуре (2) имеет самую низкую перплексией (29). Это может указывать на то, что при такой температуре генерируются более разнообразные и неожиданные тексты, но при этом снижается понятность и качество текста для обычного чтения.
Из 2 графика (см «генерация текстов разной длины») следует вывод о том, что более низкая перплексия и более короткие тексты приводят к более качественным результатам генерации.
Третий график показывает, что увеличение количества эпох обучения ведет к улучшению качества генерации текста. Но не стоит забывать о том, что это будет происходить до определенного момента. Если качество генерации перестает улучшаться или начинает ухудшаться при увеличении числа эпох, то это может свидетельствовать о том, что происходит переобучение, и дальнейшее увеличение числа эпох может быть нецелесообразным.
В данной работе мы демонстрируем обучение нейронной сети на 5 эпохах в качестве примера. Для наиболее качественной генерации текста необходимо было обучать сеть на 50 и более эпохах, но ввиду ограниченных вычислительных ресурсов и времени мы этого сделать не можем. В нашем случае обучение на 5 эпохах заняло примерно 34 минуты.
