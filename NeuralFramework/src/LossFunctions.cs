using System;

namespace NeuralFramework
{
    #region Функции потерь

    /// <summary>
    /// Базовый класс для функций потерь
    /// </summary>
    public abstract class LossFunction
    {
        public abstract double Calculate(Matrix predicted, Matrix actual);
        public abstract Matrix Gradient(Matrix predicted, Matrix actual);
    }

    /// <summary>
    /// Среднеквадратичная ошибка (MSE) - для регрессии
    /// </summary>
    public class MeanSquaredError : LossFunction
    {
        public override double Calculate(Matrix predicted, Matrix actual)
        {
            double sum = 0;
            for (int i = 0; i < predicted.Rows; i++)
                for (int j = 0; j < predicted.Cols; j++)
                {
                    double diff = predicted[i, j] - actual[i, j];
                    sum += diff * diff;
                }
            return sum / (predicted.Rows * predicted.Cols);
        }

        public override Matrix Gradient(Matrix predicted, Matrix actual)
        {
            var grad = new Matrix(predicted.Rows, predicted.Cols);
            int n = predicted.Rows * predicted.Cols;
            for (int i = 0; i < predicted.Rows; i++)
                for (int j = 0; j < predicted.Cols; j++)
                    grad[i, j] = 2 * (predicted[i, j] - actual[i, j]) / n;
            return grad;
        }
    }

    /// <summary>
    /// Кросс-энтропия - для классификации
    /// </summary>
    public class CrossEntropyLoss : LossFunction
    {
        public override double Calculate(Matrix predicted, Matrix actual)
        {
            double sum = 0;
            for (int i = 0; i < predicted.Rows; i++)
                for (int j = 0; j < predicted.Cols; j++)
                {
                    double p = Math.Max(predicted[i, j], 1e-15);
                    sum -= actual[i, j] * Math.Log(p);
                }
            return sum / predicted.Rows;
        }

        public override Matrix Gradient(Matrix predicted, Matrix actual)
        {
            var grad = new Matrix(predicted.Rows, predicted.Cols);
            for (int i = 0; i < predicted.Rows; i++)
                for (int j = 0; j < predicted.Cols; j++)
                    grad[i, j] = (predicted[i, j] - actual[i, j]) / predicted.Rows;
            return grad;
        }
    }

    /// <summary>
    /// Средняя абсолютная ошибка (MAE) - для регрессии
    /// </summary>
    public class MeanAbsoluteError : LossFunction
    {
        public override double Calculate(Matrix predicted, Matrix actual)
        {
            double sum = 0;
            for (int i = 0; i < predicted.Rows; i++)
                for (int j = 0; j < predicted.Cols; j++)
                    sum += Math.Abs(predicted[i, j] - actual[i, j]);
            return sum / (predicted.Rows * predicted.Cols);
        }

        public override Matrix Gradient(Matrix predicted, Matrix actual)
        {
            var grad = new Matrix(predicted.Rows, predicted.Cols);
            int n = predicted.Rows * predicted.Cols;
            for (int i = 0; i < predicted.Rows; i++)
                for (int j = 0; j < predicted.Cols; j++)
                {
                    double diff = predicted[i, j] - actual[i, j];
                    grad[i, j] = (diff > 0 ? 1 : -1) / n;
                }
            return grad;
        }
    }

    #endregion
}
