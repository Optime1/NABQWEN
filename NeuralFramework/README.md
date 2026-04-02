# Neural Framework - Нейросетевой фреймворк на C#

Полнофункциональный фреймворк для обучения полносвязных нейронных сетей с нуля на C#.

## Структура проекта

```
NeuralFramework/
├── NeuralFramework.csproj    # Файл проекта
├── src/
│   ├── Matrix.cs             # Матричные операции
│   ├── ActivationFunctions.cs # Функции активации
│   ├── LossFunctions.cs      # Функции потерь
│   ├── Layers.cs             # Слои нейросети
│   ├── Optimizers.cs         # Алгоритмы оптимизации
│   ├── Dataset.cs            # Работа с данными
│   └── NeuralNetwork.cs      # Нейросеть и Trainer
└── examples/
    └── Examples.cs           # Примеры использования
```

## Возможности

### Создание многослойной нейросети

```csharp
var network = new NeuralNetwork(
    new DenseLayer(784, 128, new ReLU()),
    new DenseLayer(128, 64, new ReLU()),
    new DenseLayer(64, 10, new Softmax())
);
```

### Функции активации (5 шт.)

| Класс | Описание | Применение |
|-------|----------|------------|
| `Sigmoid` | Логистическая функция σ(x) = 1/(1+e⁻ˣ) | Бинарная классификация |
| `ReLU` | Rectified Linear Unit: max(0, x) | Скрытые слои |
| `Tanh` | Гиперболический тангенс | Скрытые слои |
| `Softmax` | Нормализация в вероятности | Многоклассовая классификация |
| `Linear` | Линейная функция f(x) = x | Регрессия |

### Функции потерь (3 шт.)

| Класс | Описание | Применение |
|-------|----------|------------|
| `MeanSquaredError` | MSE = mean((y-ŷ)²) | Регрессия |
| `CrossEntropyLoss` | CE = -Σy·log(ŷ) | Классификация |
| `MeanAbsoluteError` | MAE = mean(|y-ŷ|) | Регрессия |

### Алгоритмы оптимизации (3 шт.)

Все оптимизаторы поддерживают **Gradient Clipping** для предотвращения взрыва градиентов.

| Класс | Описание | Параметры |
|-------|----------|-----------|
| `SGDOptimizer` | Стохастический градиентный спуск | learningRate, maxGradientNorm |
| `MomentumSGDOptimizer` | SGD с инерцией | learningRate, momentum, maxGradientNorm |
| `AdamOptimizer` | Adaptive Moment Estimation | learningRate, maxGradientNorm |

### Работа с данными

```csharp
// Создание датасета
var dataset = new Dataset(features, labels);

// Перемешивание
dataset = dataset.Shuffle(seed: 42);

// Мини-батчи
foreach (var (batchX, batchY) in dataset.MiniBatches(32)) { ... }

// Нормализация
dataset = dataset.Normalize();

// Разбиение train/test
var (train, test) = dataset.Split(testRatio: 0.2);

// One-hot кодирование
dataset = Dataset.OneHotEncode(features, labels, numClasses: 10);

// Преобразование данных
dataset = dataset.Map(
    featureTransform: x => x.Select(v => v / 255.0).ToArray()
);
```

### Обучение в несколько строк

```csharp
var trainer = new Trainer(network, new CrossEntropyLoss())
    .WithOptimizer(new AdamOptimizer(learningRate: 0.001))
    .WithEpochs(100)
    .WithBatchSize(32)
    .WithShuffle(true)
    .OnEpochEnd((epoch, loss) => 
        Console.WriteLine($"Epoch {epoch}: Loss = {loss}"));

trainer.Train(dataset);
```

### Оценка модели

```csharp
// Потери на тесте
double testLoss = trainer.Evaluate(testDataset);

// Точность для классификации
double accuracy = trainer.Accuracy(testDataset);

// Предсказание
double[] prediction = network.Predict(inputVector);
Matrix predictions = network.PredictBatch(inputMatrix);
```

## Примеры использования

### 1. Задача XOR

```csharp
var features = new double[][]
{
    new[] { 0.0, 0.0 }, new[] { 0.0, 1.0 },
    new[] { 1.0, 0.0 }, new[] { 1.0, 1.0 }
};
var labels = new double[][]
{
    new[] { 0.0 }, new[] { 1.0 },
    new[] { 1.0 }, new[] { 0.0 }
};

var dataset = new Dataset(features, labels);
var network = new NeuralNetwork(
    new DenseLayer(2, 4, new ReLU()),
    new DenseLayer(4, 1, new Sigmoid())
);

new Trainer(network, new MeanSquaredError())
    .WithEpochs(5000)
    .WithLearningRate(0.1)
    .Train(dataset);
```

### 2. Классификация Iris

```csharp
// Подготовка данных
var dataset = Dataset.OneHotEncode(features, labels, numClasses: 3);
var (train, test) = dataset.Split(0.3);

// Создание и обучение сети
var network = new NeuralNetwork(
    new DenseLayer(4, 8, new ReLU()),
    new DenseLayer(8, 3, new Softmax())
);

var trainer = new Trainer(network, new CrossEntropyLoss())
    .WithOptimizer(new AdamOptimizer(0.01))
    .WithEpochs(1000);

trainer.Train(train);
Console.WriteLine($"Accuracy: {trainer.Accuracy(test) * 100:F1}%");
```

### 3. Регрессия sin(x)

```csharp
// Генерация данных
for (int i = 0; i < 100; i++)
{
    double x = Random.NextDouble() * 2 * Math.PI;
    features[i] = new[] { x };
    labels[i] = new[] { Math.Sin(x) };
}

var network = new NeuralNetwork(
    new DenseLayer(1, 16, new Tanh()),
    new DenseLayer(16, 16, new Tanh()),
    new DenseLayer(16, 1, new Linear())
);

new Trainer(network, new MeanSquaredError())
    .WithOptimizer(new MomentumSGDOptimizer(0.01, 0.9))
    .WithEpochs(2000)
    .Train(dataset);
```

## Сборка и запуск

Требуется .NET SDK 8.0 или новее:

```bash
cd NeuralFramework
dotnet run
```

Или скомпилировать и запустить отдельно:

```bash
dotnet build
dotnet run --project NeuralFramework.csproj
```

## Архитектура

### Основные компоненты

1. **Matrix** - базовый класс для матричных операций
2. **Layer** - абстрактный базовый класс для слоёв
3. **DenseLayer** - полносвязный слой с весами и смещениями
4. **ActivationFunction** - базовый класс функций активации
5. **LossFunction** - базовый класс функций потерь
6. **Optimizer** - базовый класс оптимизаторов
7. **Dataset** - утилиты для работы с данными
8. **NeuralNetwork** - контейнер слоёв сети
9. **Trainer** - fluent API для конфигурирования обучения

### Прямое распространение (Forward Pass)

```
Input → [Dense + ReLU] → [Dense + ReLU] → [Dense + Softmax] → Output
```

### Обратное распространение (Backward Pass)

1. Вычисление градиента функции потерь
2. Последовательное распространение градиента назад через слои
3. Обновление весов оптимизатором

## Авторы

| Фамилия | Имя | Группа |
|---------|-----|--------|
| [Фамилия] | [Имя] | [Группа] |

## Лицензия

MIT License - свободное использование с указанием авторства.
