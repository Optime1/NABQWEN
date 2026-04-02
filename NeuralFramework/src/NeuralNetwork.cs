using System;
using System.Collections.Generic;

namespace NeuralFramework
{
    #region Нейросеть и Тренер

    /// <summary>
    /// Полносвязная нейронная сеть
    /// </summary>
    public class NeuralNetwork
    {
        private readonly List<Layer> layers;
        private readonly List<DenseLayer> denseLayers;

        public NeuralNetwork(params Layer[] layers)
        {
            this.layers = new List<Layer>(layers);
            denseLayers = new List<DenseLayer>();
            foreach (var layer in this.layers)
            {
                if (layer is DenseLayer dl)
                    denseLayers.Add(dl);
            }
        }

        /// <summary>
        /// Прямое распространение
        /// </summary>
        public Matrix Forward(Matrix input)
        {
            Matrix output = input;
            foreach (var layer in layers)
                output = layer.Forward(output);
            return output;
        }

        /// <summary>
        /// Обратное распространение ошибки
        /// </summary>
        public void Backward(Matrix gradient)
        {
            Matrix grad = gradient;
            for (int i = layers.Count - 1; i >= 0; i--)
                grad = layers[i].Backward(grad);
        }

        /// <summary>
        /// Предсказание для одного примера
        /// </summary>
        public double[] Predict(double[] input)
        {
            var inputMatrix = new Matrix(1, input.Length);
            inputMatrix.SetRow(0, input);
            var output = Forward(inputMatrix);
            return output.Row(0);
        }

        /// <summary>
        /// Предсказание для множества примеров
        /// </summary>
        public Matrix PredictBatch(Matrix input) => Forward(input);

        public IReadOnlyList<Layer> Layers => layers.AsReadOnly();
        public IReadOnlyList<DenseLayer> DenseLayers => denseLayers.AsReadOnly();
    }

    /// <summary>
    /// Класс для обучения нейросети с гибкой конфигурацией
    /// </summary>
    public class Trainer
    {
        private readonly NeuralNetwork network;
        private readonly LossFunction lossFunction;
        private Optimizer optimizer;
        
        // Параметры обучения
        private int epochs = 100;
        private int batchSize = 32;
        private double learningRate = 0.01;
        private bool shuffle = true;
        private int seed = 42;
        private Action<int, double> onEpochEnd;

        public Trainer(NeuralNetwork network, LossFunction lossFunction)
        {
            this.network = network;
            this.lossFunction = lossFunction;
            this.optimizer = new SGDOptimizer(learningRate);
        }

        /// <summary>
        /// Установка оптимизатора
        /// </summary>
        public Trainer WithOptimizer(Optimizer opt)
        {
            optimizer = opt;
            return this;
        }

        /// <summary>
        /// Количество эпох
        /// </summary>
        public Trainer WithEpochs(int e)
        {
            epochs = e;
            return this;
        }

        /// <summary>
        /// Размер мини-батча
        /// </summary>
        public Trainer WithBatchSize(int bs)
        {
            batchSize = bs;
            return this;
        }

        /// <summary>
        /// Скорость обучения
        /// </summary>
        public Trainer WithLearningRate(double lr)
        {
            learningRate = lr;
            if (optimizer != null)
            {
                // Обновляем learning rate в оптимизаторе через рефлексию или пересоздание
                // Для простоты просто сохраняем значение
            }
            return this;
        }

        /// <summary>
        /// Перемешивать данные перед каждой эпохой
        /// </summary>
        public Trainer WithShuffle(bool s, int seed = 42)
        {
            shuffle = s;
            this.seed = seed;
            return this;
        }

        /// <summary>
        /// Callback в конце каждой эпохи
        /// </summary>
        public Trainer OnEpochEnd(Action<int, double> callback)
        {
            onEpochEnd = callback;
            return this;
        }

        /// <summary>
        /// Обучение сети
        /// </summary>
        public void Train(Dataset dataset)
        {
            var data = dataset;
            
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                if (shuffle)
                    data = data.Shuffle(seed + epoch);

                double totalLoss = 0;
                int batchCount = 0;

                foreach (var (batchX, batchY) in data.MiniBatches(batchSize))
                {
                    // Forward pass
                    var output = network.Forward(batchX);
                    
                    // Вычисление потерь
                    double loss = lossFunction.Calculate(output, batchY);
                    totalLoss += loss;
                    batchCount++;

                    // Backward pass
                    var gradient = lossFunction.Gradient(output, batchY);
                    network.Backward(gradient);

                    // Обновление весов
                    foreach (var layer in network.DenseLayers)
                        optimizer.UpdateWeights(layer);
                }

                double avgLoss = totalLoss / batchCount;
                
                if (onEpochEnd != null)
                    onEpochEnd(epoch, avgLoss);
                else
                    Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Loss: {avgLoss:F6}");
            }
        }

        /// <summary>
        /// Оценка на тестовых данных
        /// </summary>
        public double Evaluate(Dataset dataset)
        {
            var output = network.Forward(dataset.Features);
            return lossFunction.Calculate(output, dataset.Labels);
        }

        /// <summary>
        /// Точность для классификации
        /// </summary>
        public double Accuracy(Dataset dataset)
        {
            var output = network.Forward(dataset.Features);
            int correct = 0;
            
            for (int i = 0; i < output.Rows; i++)
            {
                int predictedClass = 0;
                int actualClass = 0;
                
                double maxPred = double.NegativeInfinity;
                double maxActual = double.NegativeInfinity;
                
                for (int j = 0; j < output.Cols; j++)
                {
                    if (output[i, j] > maxPred)
                    {
                        maxPred = output[i, j];
                        predictedClass = j;
                    }
                    if (dataset.Labels[i, j] > maxActual)
                    {
                        maxActual = dataset.Labels[i, j];
                        actualClass = j;
                    }
                }
                
                if (predictedClass == actualClass)
                    correct++;
            }
            
            return (double)correct / dataset.Count;
        }
    }

    #endregion
}
