# Нейросетевой фреймворк на C#

Полносвязный нейросетевой фреймворк для обучения многослойных перцептронов. Реализован на C# без использования внешних библиотек.

| Фамилия Имя | 
|-------------|
| Студент     |

## Возможности

### ✅ Создание многослойной нейросети перечислением слоёв

```csharp
var network = new NeuralNetwork(
    new DenseLayer(4, 10, new ReLU()),
    new DenseLayer(10, 3, new Softmax())
);
```

### ✅ Удобные функции для работы с данными

- **Shuffle** — перемешивание датасета
- **MiniBatches** — разбивка на мини-батчи
- **Map** — трансформация данных
- **Normalize** — нормализация признаков
- **Split** — разделение на train/test

```csharp
var dataset = Dataset.FromArrays(inputs, targets);
var (train, test) = dataset.Split(0.2);
train = train.Normalize();

foreach (var (batchInputs, batchTargets) in train.MiniBatches(32))
{
    // обработка батча
}
```

### ✅ Алгоритмы оптимизации (3+)

| Оптимизатор | Описание |
|-------------|----------|
| **SGD** | Стохастический градиентный спуск |
| **MomentumSGD** | SGD с инерцией (momentum) |
| **Adam** | Адаптивный момент (по умолчанию β1=0.9, β2=0.999) |

Все оптимизаторы поддерживают **Gradient Clipping**.

```csharp
new Trainer(network, loss, new Adam())
    .EnableGradientClipping(maxNorm: 1.0)
    .Fit(trainData);
```

### ✅ Функции активации

| Функция | Класс | Для задач |
|---------|-------|-----------|
| Sigmoid | `Sigmoid` | Классификация |
| ReLU | `ReLU` | Скрытые слои |
| Tanh | `TanhActivation` | Скрытые слои |
| Softmax | `Softmax` | Многоклассовая классификация |
| Linear | `Linear` | Регрессия |

### ✅ Функции потерь

| Функция | Класс | Для задач |
|---------|-------|-----------|
| MSE (Среднеквадратичная) | `MSELoss` | Регрессия |
| CrossEntropy | `CrossEntropyLoss` | Классификация |
| MAE (Средняя абсолютная) | `MAELoss` | Регрессия |

### ✅ Обучение в несколько строк

```csharp
new Trainer(network, new CrossEntropyLoss(), new Adam())
    .SetLearningRate(0.01)
    .SetBatchSize(32)
    .SetEpochs(100)
    .EnableGradientClipping()
    .SetVerbose(true)
    .Fit(trainData, validationData);
```

## Примеры использования

### 1. Классификация Iris

```csharp
// Данные: 4 признака, 3 класса
var inputs = new double[][] { ... };
var targets = new double[][] { ... }; // one-hot encoding

var dataset = Dataset.FromArrays(inputs, targets);
var (train, test) = dataset.Split(0.3);
train = train.Normalize();

var network = new NeuralNetwork(
    new DenseLayer(4, 10, new ReLU()),
    new DenseLayer(10, 3, new Softmax())
);

new Trainer(network, new CrossEntropyLoss(), new Adam())
    .SetLearningRate(0.01)
    .SetEpochs(200)
    .Fit(train, test);
```

### 2. Регрессия (аппроксимация sin(x))

```csharp
// Генерация данных
for (int i = 0; i < 100; i++)
{
    inputs.Add(new double[] { i * 0.1 });
    targets.Add(new double[] { Math.Sin(i * 0.1) });
}

var network = new NeuralNetwork(
    new DenseLayer(1, 20, new ReLU()),
    new DenseLayer(20, 10, new ReLU()),
    new DenseLayer(10, 1, new Linear())
);

new Trainer(network, new MSELoss(), new MomentumSGD(0.9))
    .SetLearningRate(0.001)
    .SetEpochs(500)
    .EnableGradientClipping(0.5)
    .Fit(dataset);
```

### 3. XOR проблема

```csharp
var inputs = new double[][]
{
    new double[] {0, 0}, new double[] {0, 1},
    new double[] {1, 0}, new double[] {1, 1}
};

var targets = new double[][]
{
    new double[] {1, 0}, new double[] {0, 1},
    new double[] {0, 1}, new double[] {1, 0}
};

var network = new NeuralNetwork(
    new DenseLayer(2, 4, new ReLU()),
    new DenseLayer(4, 2, new Softmax())
);

new Trainer(network, new CrossEntropyLoss(), new SGD())
    .SetLearningRate(0.1)
    .SetEpochs(1000)
    .Fit(Dataset.FromArrays(inputs, targets));
```

## Структура проекта

```
NeuralFramework.cs
├── Matrix              # Матричные операции
├── IActivationFunction # Интерфейс функций активации
│   ├── Sigmoid
│   ├── ReLU
│   ├── TanhActivation
│   ├── Softmax
│   └── Linear
├── ILossFunction       # Интерфейс функций потерь
│   ├── MSELoss
│   ├── CrossEntropyLoss
│   └── MAELoss
├── ILayer              # Интерфейс слоя
│   └── DenseLayer      # Полносвязный слой
├── IOptimizer          # Интерфейс оптимизатора
│   ├── SGD
│   ├── MomentumSGD
│   └── Adam
├── NeuralNetwork       # Класс нейросети
├── Dataset             # Работа с данными
├── Trainer             # Training loop
└── Examples            # Примеры использования
```

## Компиляция и запуск

```bash
# Компиляция
csc NeuralFramework.cs

# Запуск
NeuralFramework.exe
```

Или используйте .NET CLI:

```bash
dotnet run
```

## Архитектура

Фреймворк реализует **обратное распространение ошибки (backpropagation)** для обучения сети:

1. **Forward pass**: входные данные проходят через все слои
2. **Вычисление потерь**: сравнение предсказания с целевым значением
3. **Backward pass**: вычисление градиентов от выхода к входу
4. **Обновление весов**: оптимизатор обновляет параметры сети

## Лицензия

MIT
