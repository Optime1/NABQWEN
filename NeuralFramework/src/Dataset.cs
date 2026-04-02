using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralFramework
{
    #region Работа с данными

    /// <summary>
    /// Класс для работы с датасетами
    /// </summary>
    public class Dataset
    {
        public Matrix Features { get; private set; }
        public Matrix Labels { get; private set; }
        public int Count => Features.Rows;

        public Dataset(Matrix features, Matrix labels)
        {
            Features = features;
            Labels = labels;
        }

        public Dataset(double[][] features, double[][] labels)
        {
            Features = Matrix.CreateFromRows(features);
            Labels = Matrix.CreateFromRows(labels);
        }

        /// <summary>
        /// Перемешивание данных
        /// </summary>
        public Dataset Shuffle(int seed = 42)
        {
            var rand = new Random(seed);
            var indices = Enumerable.Range(0, Count).OrderBy(x => rand.Next()).ToList();
            
            var newFeatures = new double[Count][];
            var newLabels = new double[Count][];
            
            for (int i = 0; i < Count; i++)
            {
                newFeatures[i] = Features.Row(indices[i]);
                newLabels[i] = Labels.Row(indices[i]);
            }
            
            return new Dataset(newFeatures, newLabels);
        }

        /// <summary>
        /// Разбиение на мини-батчи
        /// </summary>
        public IEnumerable<(Matrix batchX, Matrix batchY)> MiniBatches(int batchSize)
        {
            for (int i = 0; i < Count; i += batchSize)
            {
                int size = Math.Min(batchSize, Count - i);
                var batchX = new double[size][];
                var batchY = new double[size][];
                
                for (int j = 0; j < size; j++)
                {
                    batchX[j] = Features.Row(i + j);
                    batchY[j] = Labels.Row(i + j);
                }
                
                yield return (Matrix.CreateFromRows(batchX), Matrix.CreateFromRows(batchY));
            }
        }

        /// <summary>
        /// Применение функции к каждому элементу
        /// </summary>
        public Dataset Map(Func<double[], double[]> featureTransform, Func<double[], double[]> labelTransform = null)
        {
            var newFeatures = new double[Count][];
            var newLabels = new double[Count][];
            
            for (int i = 0; i < Count; i++)
            {
                newFeatures[i] = featureTransform(Features.Row(i));
                newLabels[i] = labelTransform != null ? labelTransform(Labels.Row(i)) : Labels.Row(i);
            }
            
            return new Dataset(newFeatures, newLabels);
        }

        /// <summary>
        /// Нормализация признаков (zero mean, unit variance)
        /// </summary>
        public Dataset Normalize()
        {
            var data = new double[Count][];
            for (int i = 0; i < Count; i++)
                data[i] = Features.Row(i);
            
            int nFeatures = Features.Cols;
            var means = new double[nFeatures];
            var stds = new double[nFeatures];
            
            // Вычисление среднего
            for (int j = 0; j < nFeatures; j++)
            {
                double sum = 0;
                for (int i = 0; i < Count; i++)
                    sum += data[i][j];
                means[j] = sum / Count;
            }
            
            // Вычисление стандартного отклонения
            for (int j = 0; j < nFeatures; j++)
            {
                double sumSq = 0;
                for (int i = 0; i < Count; i++)
                {
                    double diff = data[i][j] - means[j];
                    sumSq += diff * diff;
                }
                stds[j] = Math.Sqrt(sumSq / Count);
                if (stds[j] < 1e-8) stds[j] = 1;
            }
            
            // Нормализация
            var normalized = new double[Count][];
            for (int i = 0; i < Count; i++)
            {
                normalized[i] = new double[nFeatures];
                for (int j = 0; j < nFeatures; j++)
                    normalized[i][j] = (data[i][j] - means[j]) / stds[j];
            }
            
            var labelsArray = new double[Count][];
            for (int i = 0; i < Count; i++)
                labelsArray[i] = Labels.Row(i);
            
            return new Dataset(normalized, labelsArray);
        }

        /// <summary>
        /// Разбиение на train/test
        /// </summary>
        public (Dataset train, Dataset test) Split(double testRatio = 0.2, int seed = 42)
        {
            var shuffled = Shuffle(seed);
            int testCount = (int)(Count * testRatio);
            int trainCount = Count - testCount;
            
            var trainX = new double[trainCount][];
            var trainY = new double[trainCount][];
            var testX = new double[testCount][];
            var testY = new double[testCount][];
            
            for (int i = 0; i < trainCount; i++)
            {
                trainX[i] = shuffled.Features.Row(i);
                trainY[i] = shuffled.Labels.Row(i);
            }
            
            for (int i = 0; i < testCount; i++)
            {
                testX[i] = shuffled.Features.Row(trainCount + i);
                testY[i] = shuffled.Labels.Row(trainCount + i);
            }
            
            return (new Dataset(trainX, trainY), new Dataset(testX, testY));
        }

        /// <summary>
        /// One-hot кодирование меток
        /// </summary>
        public static Dataset OneHotEncode(double[][] features, int[] labels, int numClasses)
        {
            var oneHot = new double[labels.Length][];
            for (int i = 0; i < labels.Length; i++)
            {
                oneHot[i] = new double[numClasses];
                oneHot[i][labels[i]] = 1.0;
            }
            return new Dataset(features, oneHot);
        }
    }

    #endregion
}
