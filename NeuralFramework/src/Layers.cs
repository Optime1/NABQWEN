using System;
using System.Collections.Generic;

namespace NeuralFramework
{
    #region Слои нейросети

    /// <summary>
    /// Базовый класс для слоёв нейросети
    /// </summary>
    public abstract class Layer
    {
        public Matrix Output { get; protected set; }
        public Matrix Input { get; protected set; }
        
        public abstract Matrix Forward(Matrix input);
        public abstract Matrix Backward(Matrix gradient);
        public abstract void UpdateWeights(double learningRate);
    }

    /// <summary>
    /// Полносвязный слой (Dense Layer)
    /// </summary>
    public class DenseLayer : Layer
    {
        private Matrix weights;
        private Matrix biases;
        private Matrix weightGradients;
        private Matrix biasGradients;
        private readonly ActivationFunction activation;
        private readonly bool useBias;

        public int InputSize { get; }
        public int OutputSize { get; }

        public DenseLayer(int inputSize, int outputSize, ActivationFunction activation = null, bool useBias = true)
        {
            InputSize = inputSize;
            OutputSize = outputSize;
            this.activation = activation ?? new ReLU();
            this.useBias = useBias;

            // Инициализация весов методом Xavier/Glorot
            double scale = Math.Sqrt(2.0 / (inputSize + outputSize));
            weights = Matrix.Random(outputSize, inputSize, scale);
            biases = Matrix.Random(outputSize, 1, scale);
            weightGradients = Matrix.Zeros(outputSize, inputSize);
            biasGradients = Matrix.Zeros(outputSize, 1);
        }

        public override Matrix Forward(Matrix input)
        {
            Input = input;
            // Z = W * X + b
            var z = weights * input;
            
            if (useBias)
            {
                for (int i = 0; i < OutputSize; i++)
                    for (int j = 0; j < input.Cols; j++)
                        z[i, j] += biases[i, 0];
            }

            // Применяем функцию активации
            Output = activation.Apply(z);
            return Output;
        }

        public override Matrix Backward(Matrix gradient)
        {
            // Производная функции активации
            var activationGrad = activation.ApplyDerivative(Output);
            
            // Градиент перед активацией
            var preActivationGrad = new Matrix(Output.Rows, Output.Cols);
            for (int i = 0; i < Output.Rows; i++)
                for (int j = 0; j < Output.Cols; j++)
                    preActivationGrad[i, j] = gradient[i, j] * activationGrad[i, j];

            // Градиент весов: dW = grad * X^T
            weightGradients = preActivationGrad * Input.Transpose();

            // Градиент смещений
            if (useBias)
            {
                biasGradients = Matrix.Zeros(OutputSize, 1);
                for (int i = 0; i < OutputSize; i++)
                    for (int j = 0; j < preActivationGrad.Cols; j++)
                        biasGradients[i, 0] += preActivationGrad[i, j];
            }

            // Градиент для предыдущего слоя: dX = W^T * grad
            return weights.Transpose() * preActivationGrad;
        }

        public override void UpdateWeights(double learningRate)
        {
            weights = weights - learningRate * weightGradients;
            if (useBias)
                biases = biases - learningRate * biasGradients;
        }

        public Matrix Weights => weights;
        public Matrix Biases => biases;
        
        // Свойства для доступа к градиентам из оптимизатора
        internal Matrix WeightGradients => weightGradients;
        internal Matrix BiasGradients => biasGradients;
        
        // Метод для обновления весов с новыми значениями градиентов
        internal void ApplyUpdate(Matrix newWeights, Matrix newBiases)
        {
            weights = newWeights;
            biases = newBiases;
        }
    }

    /// <summary>
    /// Слой активации (отдельный слой)
    /// </summary>
    public class ActivationLayer : Layer
    {
        private readonly ActivationFunction activation;

        public ActivationLayer(ActivationFunction activation)
        {
            this.activation = activation;
        }

        public override Matrix Forward(Matrix input)
        {
            Input = input;
            Output = activation.Apply(input);
            return Output;
        }

        public override Matrix Backward(Matrix gradient)
        {
            var activationGrad = activation.ApplyDerivative(Input);
            var result = new Matrix(gradient.Rows, gradient.Cols);
            for (int i = 0; i < gradient.Rows; i++)
                for (int j = 0; j < gradient.Cols; j++)
                    result[i, j] = gradient[i, j] * activationGrad[i, j];
            return result;
        }

        public override void UpdateWeights(double learningRate) { }
    }

    #endregion
}
