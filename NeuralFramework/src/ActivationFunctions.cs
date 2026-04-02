using System;

namespace NeuralFramework
{
    #region Функции активации

    /// <summary>
    /// Базовый класс для функций активации
    /// </summary>
    public abstract class ActivationFunction
    {
        public virtual bool IsElementWise => true;
        public virtual Matrix Apply(Matrix m) => m.Apply(Activate);
        public virtual Matrix ApplyDerivative(Matrix m) => m.Apply(Derivative);
        public virtual double Activate(double x) => throw new NotImplementedException();
        public virtual double Derivative(double x) => throw new NotImplementedException();
    }

    /// <summary>
    /// Сигмоида (логистическая функция)
    /// </summary>
    public class Sigmoid : ActivationFunction
    {
        public override double Activate(double x) => 1.0 / (1.0 + Math.Exp(-x));
        public override double Derivative(double x)
        {
            var s = Activate(x);
            return s * (1 - s);
        }
    }

    /// <summary>
    /// ReLU (Rectified Linear Unit)
    /// </summary>
    public class ReLU : ActivationFunction
    {
        public override double Activate(double x) => Math.Max(0, x);
        public override double Derivative(double x) => x > 0 ? 1 : 0;
    }

    /// <summary>
    /// Гиперболический тангенс
    /// </summary>
    public class Tanh : ActivationFunction
    {
        public override double Activate(double x) => Math.Tanh(x);
        public override double Derivative(double x)
        {
            var t = Math.Tanh(x);
            return 1 - t * t;
        }
    }

    /// <summary>
    /// Softmax (для многоклассовой классификации)
    /// </summary>
    public class Softmax : ActivationFunction
    {
        public override bool IsElementWise => false;

        public override Matrix Apply(Matrix m)
        {
            var result = new Matrix(m.Rows, m.Cols);
            for (int i = 0; i < m.Rows; i++)
            {
                double max = double.NegativeInfinity;
                for (int j = 0; j < m.Cols; j++)
                    max = Math.Max(max, m[i, j]);

                double sum = 0;
                for (int j = 0; j < m.Cols; j++)
                {
                    result[i, j] = Math.Exp(m[i, j] - max);
                    sum += result[i, j];
                }
                for (int j = 0; j < m.Cols; j++)
                    result[i, j] /= sum;
            }
            return result;
        }

        public override Matrix ApplyDerivative(Matrix output)
        {
            // Для комбинации с CrossEntropy градиент упрощается
            // Возвращаем output * (1 - output) для каждого элемента
            var result = new Matrix(output.Rows, output.Cols);
            for (int i = 0; i < output.Rows; i++)
                for (int j = 0; j < output.Cols; j++)
                    result[i, j] = output[i, j] * (1.0 - output[i, j]);
            return result;
        }
    }

    /// <summary>
    /// Линейная функция (для регрессии)
    /// </summary>
    public class Linear : ActivationFunction
    {
        public override double Activate(double x) => x;
        public override double Derivative(double x) => 1;
    }

    #endregion
}
