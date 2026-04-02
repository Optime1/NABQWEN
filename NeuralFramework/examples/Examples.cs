using System;

namespace NeuralFramework.Examples
{
    /// <summary>
    /// Пример 1: Решение задачи XOR
    /// </summary>
    public class XorExample
    {
        public static void Run()
        {
            Console.WriteLine("=== Пример 1: XOR ===\n");

            // Данные XOR
            var features = new double[][]
            {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 }
            };

            var labels = new double[][]
            {
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 0 }
            };

            var dataset = new Dataset(features, labels);

            // Создание сети: 2 -> 4 (ReLU) -> 1 (Sigmoid)
            var network = new NeuralNetwork(
                new DenseLayer(2, 4, new ReLU()),
                new DenseLayer(4, 1, new Sigmoid())
            );

            // Обучение в несколько строк
            var trainer = new Trainer(network, new MeanSquaredError())
                .WithEpochs(5000)
                .WithBatchSize(4)
                .WithLearningRate(0.1)
                .WithShuffle(true);

            trainer.Train(dataset);

            // Проверка результатов
            Console.WriteLine("\nРезультаты предсказания:");
            for (int i = 0; i < 4; i++)
            {
                var prediction = network.Predict(features[i]);
                Console.WriteLine($"XOR({features[i][0]}, {features[i][1]}) = {prediction[0]:F4} (ожидается: {labels[i][0]})");
            }
        }
    }

    /// <summary>
    /// Пример 2: Классификация Iris
    /// </summary>
    public class IrisExample
    {
        public static void Run()
        {
            Console.WriteLine("\n=== Пример 2: Iris Classification ===\n");

            // Упрощённый датасет Iris (15 образцов для демонстрации)
            // В реальности нужно загружать полный датасет
            var features = new double[][]
            {
                // Setosa
                new double[] { 5.1, 3.5, 1.4, 0.2 },
                new double[] { 4.9, 3.0, 1.4, 0.2 },
                new double[] { 4.7, 3.2, 1.3, 0.2 },
                new double[] { 5.0, 3.6, 1.4, 0.2 },
                new double[] { 5.4, 3.9, 1.7, 0.4 },
                // Versicolor
                new double[] { 7.0, 3.2, 4.7, 1.4 },
                new double[] { 6.4, 3.2, 4.5, 1.5 },
                new double[] { 6.9, 3.1, 4.9, 1.5 },
                new double[] { 5.5, 2.3, 4.0, 1.3 },
                new double[] { 6.5, 2.8, 4.6, 1.5 },
                // Virginica
                new double[] { 6.3, 3.3, 6.0, 2.5 },
                new double[] { 5.8, 2.7, 5.1, 1.9 },
                new double[] { 7.1, 3.0, 5.9, 2.1 },
                new double[] { 6.3, 2.9, 5.6, 1.8 },
                new double[] { 6.5, 3.0, 5.8, 2.2 }
            };

            int[] labels = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 };

            // One-hot кодирование
            var dataset = Dataset.OneHotEncode(features, labels, 3);

            // Разбиение на train/test
            var (trainData, testData) = dataset.Split(testRatio: 0.3);

            // Создание сети: 4 -> 8 (ReLU) -> 3 (Softmax)
            var network = new NeuralNetwork(
                new DenseLayer(4, 8, new ReLU()),
                new DenseLayer(8, 3, new Softmax())
            );

            // Обучение с Adam оптимизатором
            var trainer = new Trainer(network, new CrossEntropyLoss())
                .WithOptimizer(new AdamOptimizer(learningRate: 0.01))
                .WithEpochs(1000)
                .WithBatchSize(5)
                .WithShuffle(true)
                .OnEpochEnd((epoch, loss) =>
                {
                    if ((epoch + 1) % 200 == 0)
                        Console.WriteLine($"Epoch {epoch + 1}/1000, Loss: {loss:F4}");
                });

            trainer.Train(trainData);

            // Оценка точности
            double trainAccuracy = trainer.Accuracy(trainData);
            double testAccuracy = trainer.Accuracy(testData);

            Console.WriteLine($"\nTrain Accuracy: {trainAccuracy * 100:F1}%");
            Console.WriteLine($"Test Accuracy: {testAccuracy * 100:F1}%");

            // Предсказания
            Console.WriteLine("\nПримеры предсказаний:");
            for (int i = 0; i < Math.Min(5, testData.Count); i++)
            {
                var prediction = network.Predict(testData.Features.Row(i));
                int predictedClass = 0;
                double maxVal = double.NegativeInfinity;
                for (int j = 0; j < 3; j++)
                {
                    if (prediction[j] > maxVal)
                    {
                        maxVal = prediction[j];
                        predictedClass = j;
                    }
                }

                string[] classNames = { "Setosa", "Versicolor", "Virginica" };
                
                int actualClass = 0;
                maxVal = double.NegativeInfinity;
                var actualLabelRow = testData.Labels.Row(i);
                for (int j = 0; j < 3; j++)
                {
                    if (actualLabelRow[j] > maxVal)
                    {
                        maxVal = actualLabelRow[j];
                        actualClass = j;
                    }
                }

                Console.WriteLine($"Предсказано: {classNames[predictedClass]}, Фактически: {classNames[actualClass]}");
            }
        }
    }

    /// <summary>
    /// Пример 3: Регрессия sin(x)
    /// </summary>
    public class RegressionExample
    {
        public static void Run()
        {
            Console.WriteLine("\n=== Пример 3: Регрессия sin(x) ===\n");

            // Генерация данных
            var rand = new Random(42);
            var features = new double[100][];
            var labels = new double[100][];

            for (int i = 0; i < 100; i++)
            {
                double x = rand.NextDouble() * 2 * Math.PI;
                features[i] = new double[] { x };
                labels[i] = new double[] { Math.Sin(x) };
            }

            var dataset = new Dataset(features, labels);
            var (trainData, testData) = dataset.Split(testRatio: 0.2);

            // Нормализация признаков
            trainData = trainData.Normalize();
            
            // Пересоздаём тест с теми же параметрами нормализации (для простоты используем как есть)
            testData = testData.Normalize();

            // Создание сети: 1 -> 16 (Tanh) -> 16 (Tanh) -> 1 (Linear)
            var network = new NeuralNetwork(
                new DenseLayer(1, 16, new Tanh()),
                new DenseLayer(16, 16, new Tanh()),
                new DenseLayer(16, 1, new Linear())
            );

            // Обучение с Momentum SGD
            var trainer = new Trainer(network, new MeanSquaredError())
                .WithOptimizer(new MomentumSGDOptimizer(learningRate: 0.01, momentum: 0.9))
                .WithEpochs(2000)
                .WithBatchSize(16)
                .WithShuffle(true)
                .OnEpochEnd((epoch, loss) =>
                {
                    if ((epoch + 1) % 500 == 0)
                        Console.WriteLine($"Epoch {epoch + 1}/2000, Loss: {loss:F6}");
                });

            trainer.Train(trainData);

            // Оценка
            double trainLoss = trainer.Evaluate(trainData);
            double testLoss = trainer.Evaluate(testData);

            Console.WriteLine($"\nTrain MSE: {trainLoss:F6}");
            Console.WriteLine($"Test MSE: {testLoss:F6}");

            // Примеры предсказаний
            Console.WriteLine("\nПримеры предсказаний:");
            int samplesToShow = Math.Min(5, testData.Count);
            for (int i = 0; i < samplesToShow; i++)
            {
                var input = testData.Features.Row(i);
                var prediction = network.Predict(input);
                double x = input[0]; // Это нормализованное значение
                
                Console.WriteLine($"sin(x): предсказано = {prediction[0]:F4}, фактически = {testData.Labels.Row(i)[0]:F4}");
            }
        }
    }

    /// <summary>
    /// Главная программа для запуска всех примеров
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("========================================");
            Console.WriteLine("  Neural Framework - Примеры использования");
            Console.WriteLine("========================================\n");

            XorExample.Run();
            IrisExample.Run();
            RegressionExample.Run();

            Console.WriteLine("\n========================================");
            Console.WriteLine("  Все примеры завершены!");
            Console.WriteLine("========================================");
        }
    }
}
