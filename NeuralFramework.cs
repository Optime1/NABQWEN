using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;

namespace NeuralFramework
{
    #region Математические утилиты

    /// <summary>
    /// Класс для работы с матрицами и векторами
    /// </summary>
    public class Matrix
    {
        public double[,] Data { get; private set; }
        public int Rows => Data.GetLength(0);
        public int Cols => Data.GetLength(1);

        public Matrix(int rows, int cols)
        {
            Data = new double[rows, cols];
        }

        public Matrix(double[,] data)
        {
            Data = (double[,])data.Clone();
        }

        public static Matrix CreateFromRows(List<double[]> rows)
        {
            int r = rows.Count;
            int c = rows[0].Length;
            var m = new Matrix(r, c);
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    m.Data[i, j] = rows[i][j];
            return m;
        }

        public static Matrix Zeros(int rows, int cols) => new Matrix(rows, cols);

        public static Matrix Ones(int rows, int cols)
        {
            var m = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    m.Data[i, j] = 1.0;
            return m;
        }

        public static Matrix Random(int rows, int cols, double scale = 0.1)
        {
            var rand = new Random();
            var m = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    m.Data[i, j] = (rand.NextDouble() * 2 - 1) * scale;
            return m;
        }

        public double this[int i, int j]
        {
            get => Data[i, j];
            set => Data[i, j] = value;
        }

        public Matrix Copy() => new Matrix(Data);

        public static Matrix operator +(Matrix a, Matrix b)
        {
            var result = new Matrix(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    result.Data[i, j] = a.Data[i, j] + b.Data[i, j];
            return result;
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            var result = new Matrix(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    result.Data[i, j] = a.Data[i, j] - b.Data[i, j];
            return result;
        }

        public static Matrix operator *(Matrix a, Matrix b)
        {
            var result = new Matrix(a.Rows, b.Cols);
            for (int i = 0; i < a.Rows; i++)
                for (int k = 0; k < a.Cols; k++)
                    for (int j = 0; j < b.Cols; j++)
                        result.Data[i, j] += a.Data[i, k] * b.Data[k, j];
            return result;
        }

        public static Matrix operator *(Matrix a, double scalar)
        {
            var result = new Matrix(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    result.Data[i, j] = a.Data[i, j] * scalar;
            return result;
        }

        public static Matrix operator /(Matrix a, double scalar)
        {
            var result = new Matrix(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    result.Data[i, j] = a.Data[i, j] / scalar;
            return result;
        }

        // Поэлементное умножение
        public Matrix Hadamard(Matrix other)
        {
            var result = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result.Data[i, j] = Data[i, j] * other.Data[i, j];
            return result;
        }

        // Сумма по строкам (для градиентов смещений)
        public Matrix SumOverRows()
        {
            var result = new Matrix(1, Cols);
            for (int j = 0; j < Cols; j++)
                for (int i = 0; i < Rows; i++)
                    result.Data[0, j] += Data[i, j];
            return result;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Matrix[{Rows}x{Cols}]");
            for (int i = 0; i < Math.Min(Rows, 5); i++)
            {
                sb.Append("[");
                for (int j = 0; j < Math.Min(Cols, 10); j++)
                    sb.Append($"{Data[i, j]:F4}, ");
                sb.AppendLine("...]");
            }
            return sb.ToString();
        }
    }

    #endregion

    #region Функции активации

    public interface IActivationFunction
    {
        Matrix Forward(Matrix input);
        Matrix Backward(Matrix input, Matrix gradOutput);
        string Name { get; }
    }

    public class Sigmoid : IActivationFunction
    {
        public string Name => "Sigmoid";

        public Matrix Forward(Matrix input)
        {
            var output = new Matrix(input.Rows, input.Cols);
            for (int i = 0; i < input.Rows; i++)
                for (int j = 0; j < input.Cols; j++)
                    output[i, j] = 1.0 / (1.0 + Math.Exp(-input[i, j]));
            return output;
        }

        public Matrix Backward(Matrix input, Matrix gradOutput)
        {
            var sigmoid = Forward(input);
            var grad = new Matrix(input.Rows, input.Cols);
            for (int i = 0; i < input.Rows; i++)
                for (int j = 0; j < input.Cols; j++)
                    grad[i, j] = sigmoid[i, j] * (1 - sigmoid[i, j]) * gradOutput[i, j];
            return grad;
        }
    }

    public class ReLU : IActivationFunction
    {
        public string Name => "ReLU";

        public Matrix Forward(Matrix input)
        {
            var output = new Matrix(input.Rows, input.Cols);
            for (int i = 0; i < input.Rows; i++)
                for (int j = 0; j < input.Cols; j++)
                    output[i, j] = Math.Max(0, input[i, j]);
            return output;
        }

        public Matrix Backward(Matrix input, Matrix gradOutput)
        {
            var grad = new Matrix(input.Rows, input.Cols);
            for (int i = 0; i < input.Rows; i++)
                for (int j = 0; j < input.Cols; j++)
                    grad[i, j] = (input[i, j] > 0 ? 1 : 0) * gradOutput[i, j];
            return grad;
        }
    }

    public class TanhActivation : IActivationFunction
    {
        public string Name => "Tanh";

        public Matrix Forward(Matrix input)
        {
            var output = new Matrix(input.Rows, input.Cols);
            for (int i = 0; i < input.Rows; i++)
                for (int j = 0; j < input.Cols; j++)
                    output[i, j] = Math.Tanh(input[i, j]);
            return output;
        }

        public Matrix Backward(Matrix input, Matrix gradOutput)
        {
            var grad = new Matrix(input.Rows, input.Cols);
            for (int i = 0; i < input.Rows; i++)
                for (int j = 0; j < input.Cols; j++)
                {
                    var tanh = Math.Tanh(input[i, j]);
                    grad[i, j] = (1 - tanh * tanh) * gradOutput[i, j];
                }
            return grad;
        }
    }

    public class Softmax : IActivationFunction
    {
        public string Name => "Softmax";

        public Matrix Forward(Matrix input)
        {
            var output = new Matrix(input.Rows, input.Cols);
            for (int i = 0; i < input.Rows; i++)
            {
                double max = double.NegativeInfinity;
                for (int j = 0; j < input.Cols; j++)
                    max = Math.Max(max, input[i, j]);

                double sum = 0;
                for (int j = 0; j < input.Cols; j++)
                {
                    output[i, j] = Math.Exp(input[i, j] - max);
                    sum += output[i, j];
                }
                for (int j = 0; j < input.Cols; j++)
                    output[i, j] /= sum;
            }
            return output;
        }

        public Matrix Backward(Matrix input, Matrix gradOutput)
        {
            // Упрощённая версия для использования с CrossEntropyLoss
            return gradOutput;
        }
    }

    public class Linear : IActivationFunction
    {
        public string Name => "Linear";

        public Matrix Forward(Matrix input) => input.Copy();

        public Matrix Backward(Matrix input, Matrix gradOutput) => gradOutput.Copy();
    }

    #endregion

    #region Функции потерь

    public interface ILossFunction
    {
        double Forward(Matrix predictions, Matrix targets);
        Matrix Backward(Matrix predictions, Matrix targets);
        string Name { get; }
    }

    public class MSELoss : ILossFunction
    {
        public string Name => "MSE";

        public double Forward(Matrix predictions, Matrix targets)
        {
            double sum = 0;
            for (int i = 0; i < predictions.Rows; i++)
                for (int j = 0; j < predictions.Cols; j++)
                {
                    double diff = predictions[i, j] - targets[i, j];
                    sum += diff * diff;
                }
            return sum / predictions.Rows;
        }

        public Matrix Backward(Matrix predictions, Matrix targets)
        {
            var grad = new Matrix(predictions.Rows, predictions.Cols);
            for (int i = 0; i < predictions.Rows; i++)
                for (int j = 0; j < predictions.Cols; j++)
                    grad[i, j] = 2 * (predictions[i, j] - targets[i, j]) / predictions.Rows;
            return grad;
        }
    }

    public class CrossEntropyLoss : ILossFunction
    {
        public string Name => "CrossEntropy";

        public double Forward(Matrix predictions, Matrix targets)
        {
            double loss = 0;
            for (int i = 0; i < predictions.Rows; i++)
            {
                for (int j = 0; j < predictions.Cols; j++)
                {
                    if (targets[i, j] > 0)
                    {
                        double p = Math.Max(predictions[i, j], 1e-10);
                        loss -= targets[i, j] * Math.Log(p);
                    }
                }
            }
            return loss / predictions.Rows;
        }

        public Matrix Backward(Matrix predictions, Matrix targets)
        {
            var grad = new Matrix(predictions.Rows, predictions.Cols);
            for (int i = 0; i < predictions.Rows; i++)
                for (int j = 0; j < predictions.Cols; j++)
                    grad[i, j] = (predictions[i, j] - targets[i, j]) / predictions.Rows;
            return grad;
        }
    }

    public class MAELoss : ILossFunction
    {
        public string Name => "MAE";

        public double Forward(Matrix predictions, Matrix targets)
        {
            double sum = 0;
            for (int i = 0; i < predictions.Rows; i++)
                for (int j = 0; j < predictions.Cols; j++)
                    sum += Math.Abs(predictions[i, j] - targets[i, j]);
            return sum / predictions.Rows;
        }

        public Matrix Backward(Matrix predictions, Matrix targets)
        {
            var grad = new Matrix(predictions.Rows, predictions.Cols);
            for (int i = 0; i < predictions.Rows; i++)
                for (int j = 0; j < predictions.Cols; j++)
                    grad[i, j] = Math.Sign(predictions[i, j] - targets[i, j]) / predictions.Rows;
            return grad;
        }
    }

    #endregion

    #region Слои нейросети

    public interface ILayer
    {
        Matrix Forward(Matrix input, bool training = true);
        Matrix Backward(Matrix gradOutput);
        List<(Matrix weights, Matrix bias, Matrix gradWeights, Matrix gradBias)> GetParameters();
        void SetOptimizerState(IOptimizer optimizer);
        string Name { get; }
    }

    public class DenseLayer : ILayer
    {
        private Matrix weights;
        private Matrix bias;
        private Matrix gradWeights;
        private Matrix gradBias;
        private Matrix lastInput;
        private readonly IActivationFunction activation;
        private readonly Random rand = new Random();

        public string Name => $"Dense({weights.Cols})";

        public DenseLayer(int inputSize, int outputSize, IActivationFunction activation = null)
        {
            // Инициализация весов по Xavier
            double scale = Math.Sqrt(2.0 / (inputSize + outputSize));
            weights = Matrix.Random(outputSize, inputSize, scale);
            bias = Matrix.Zeros(1, outputSize);
            gradWeights = Matrix.Zeros(outputSize, inputSize);
            gradBias = Matrix.Zeros(1, outputSize);
            this.activation = activation ?? new Linear();
        }

        public Matrix Forward(Matrix input, bool training = true)
        {
            lastInput = input.Copy();
            // Z = X * W^T + b
            var z = new Matrix(input.Rows, weights.Rows);
            for (int i = 0; i < input.Rows; i++)
            {
                for (int j = 0; j < weights.Rows; j++)
                {
                    double sum = bias[0, j];
                    for (int k = 0; k < input.Cols; k++)
                        sum += input[i, k] * weights[j, k];
                    z[i, j] = sum;
                }
            }
            return activation.Forward(z);
        }

        public Matrix Backward(Matrix gradOutput)
        {
            // Градиент после функции активации
            var z = new Matrix(lastInput.Rows, weights.Rows);
            for (int i = 0; i < lastInput.Rows; i++)
            {
                for (int j = 0; j < weights.Rows; j++)
                {
                    double sum = bias[0, j];
                    for (int k = 0; k < lastInput.Cols; k++)
                        sum += lastInput[i, k] * weights[j, k];
                    z[i, j] = sum;
                }
            }
            var gradAfterActivation = activation.Backward(z, gradOutput);

            // Градиент весов: dW = grad^T * X
            gradWeights = Matrix.Zeros(weights.Rows, weights.Cols);
            for (int j = 0; j < weights.Rows; j++)
            {
                for (int k = 0; k < weights.Cols; k++)
                {
                    for (int i = 0; i < gradAfterActivation.Rows; i++)
                        gradWeights[j, k] += gradAfterActivation[i, j] * lastInput[i, k];
                }
            }

            // Градиент смещения
            gradBias = gradAfterActivation.SumOverRows();

            // Градиент для предыдущего слоя: dX = grad * W
            var gradInput = new Matrix(lastInput.Rows, lastInput.Cols);
            for (int i = 0; i < lastInput.Rows; i++)
            {
                for (int k = 0; k < lastInput.Cols; k++)
                {
                    for (int j = 0; j < weights.Rows; j++)
                        gradInput[i, k] += gradAfterActivation[i, j] * weights[j, k];
                }
            }

            return gradInput;
        }

        public List<(Matrix weights, Matrix bias, Matrix gradWeights, Matrix gradBias)> GetParameters()
        {
            return new List<(Matrix, Matrix, Matrix, Matrix)> { (weights, bias, gradWeights, gradBias) };
        }

        public void SetOptimizerState(IOptimizer optimizer)
        {
            optimizer.RegisterParameter(weights, gradWeights);
            optimizer.RegisterParameter(bias, gradBias);
        }
    }

    #endregion

    #region Оптимизаторы

    public interface IOptimizer
    {
        void RegisterParameter(Matrix param, Matrix grad);
        void Step(double learningRate);
        void ResetGradients();
        void ClipGradients(double maxNorm);
        string Name { get; }
    }

    public class SGD : IOptimizer
    {
        private List<(Matrix param, Matrix grad)> parameters = new();
        public string Name => "SGD";

        public void RegisterParameter(Matrix param, Matrix grad)
        {
            parameters.Add((param, grad));
        }

        public void Step(double learningRate)
        {
            foreach (var (param, grad) in parameters)
            {
                for (int i = 0; i < param.Rows; i++)
                    for (int j = 0; j < param.Cols; j++)
                        param[i, j] -= learningRate * grad[i, j];
            }
        }

        public void ResetGradients()
        {
            foreach (var (_, grad) in parameters)
            {
                for (int i = 0; i < grad.Rows; i++)
                    for (int j = 0; j < grad.Cols; j++)
                        grad[i, j] = 0;
            }
        }

        public void ClipGradients(double maxNorm)
        {
            // Вычисление общей нормы градиентов
            double totalNorm = 0;
            foreach (var (_, grad) in parameters)
            {
                for (int i = 0; i < grad.Rows; i++)
                    for (int j = 0; j < grad.Cols; j++)
                        totalNorm += grad[i, j] * grad[i, j];
            }
            totalNorm = Math.Sqrt(totalNorm);

            if (totalNorm > maxNorm)
            {
                double scale = maxNorm / totalNorm;
                foreach (var (_, grad) in parameters)
                {
                    for (int i = 0; i < grad.Rows; i++)
                        for (int j = 0; j < grad.Cols; j++)
                            grad[i, j] *= scale;
                }
            }
        }
    }

    public class MomentumSGD : IOptimizer
    {
        private List<(Matrix param, Matrix grad, Matrix velocity)> parameters = new();
        private readonly double momentum;
        public string Name => "MomentumSGD";

        public MomentumSGD(double momentum = 0.9)
        {
            this.momentum = momentum;
        }

        public void RegisterParameter(Matrix param, Matrix grad)
        {
            var velocity = Matrix.Zeros(param.Rows, param.Cols);
            parameters.Add((param, grad, velocity));
        }

        public void Step(double learningRate)
        {
            foreach (var (param, grad, velocity) in parameters)
            {
                for (int i = 0; i < param.Rows; i++)
                {
                    for (int j = 0; j < param.Cols; j++)
                    {
                        velocity[i, j] = momentum * velocity[i, j] + learningRate * grad[i, j];
                        param[i, j] -= velocity[i, j];
                    }
                }
            }
        }

        public void ResetGradients()
        {
            foreach (var (_, grad, _) in parameters)
            {
                for (int i = 0; i < grad.Rows; i++)
                    for (int j = 0; j < grad.Cols; j++)
                        grad[i, j] = 0;
            }
        }

        public void ClipGradients(double maxNorm)
        {
            double totalNorm = 0;
            foreach (var (_, grad, _) in parameters)
            {
                for (int i = 0; i < grad.Rows; i++)
                    for (int j = 0; j < grad.Cols; j++)
                        totalNorm += grad[i, j] * grad[i, j];
            }
            totalNorm = Math.Sqrt(totalNorm);

            if (totalNorm > maxNorm)
            {
                double scale = maxNorm / totalNorm;
                foreach (var (_, grad, _) in parameters)
                {
                    for (int i = 0; i < grad.Rows; i++)
                        for (int j = 0; j < grad.Cols; j++)
                            grad[i, j] *= scale;
                }
            }
        }
    }

    public class Adam : IOptimizer
    {
        private List<(Matrix param, Matrix grad, Matrix m, Matrix v)> parameters = new();
        private readonly double beta1, beta2, epsilon;
        private int t = 0;
        public string Name => "Adam";

        public Adam(double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
        }

        public void RegisterParameter(Matrix param, Matrix grad)
        {
            var m = Matrix.Zeros(param.Rows, param.Cols);
            var v = Matrix.Zeros(param.Rows, param.Cols);
            parameters.Add((param, grad, m, v));
        }

        public void Step(double learningRate)
        {
            t++;
            double biasCorrection1 = 1 - Math.Pow(beta1, t);
            double biasCorrection2 = 1 - Math.Pow(beta2, t);

            foreach (var (param, grad, m, v) in parameters)
            {
                for (int i = 0; i < param.Rows; i++)
                {
                    for (int j = 0; j < param.Cols; j++)
                    {
                        m[i, j] = beta1 * m[i, j] + (1 - beta1) * grad[i, j];
                        v[i, j] = beta2 * v[i, j] + (1 - beta2) * grad[i, j] * grad[i, j];
                        
                        double mHat = m[i, j] / biasCorrection1;
                        double vHat = v[i, j] / biasCorrection2;
                        
                        param[i, j] -= learningRate * mHat / (Math.Sqrt(vHat) + epsilon);
                    }
                }
            }
        }

        public void ResetGradients()
        {
            foreach (var (_, grad, _, _) in parameters)
            {
                for (int i = 0; i < grad.Rows; i++)
                    for (int j = 0; j < grad.Cols; j++)
                        grad[i, j] = 0;
            }
        }

        public void ClipGradients(double maxNorm)
        {
            double totalNorm = 0;
            foreach (var (_, grad, _, _) in parameters)
            {
                for (int i = 0; i < grad.Rows; i++)
                    for (int j = 0; j < grad.Cols; j++)
                        totalNorm += grad[i, j] * grad[i, j];
            }
            totalNorm = Math.Sqrt(totalNorm);

            if (totalNorm > maxNorm)
            {
                double scale = maxNorm / totalNorm;
                foreach (var (_, grad, _, _) in parameters)
                {
                    for (int i = 0; i < grad.Rows; i++)
                        for (int j = 0; j < grad.Cols; j++)
                            grad[i, j] *= scale;
                }
            }
        }
    }

    #endregion

    #region Нейросеть

    public class NeuralNetwork
    {
        private readonly List<ILayer> layers;
        private IOptimizer optimizer;

        public NeuralNetwork(params ILayer[] layers)
        {
            this.layers = new List<ILayer>(layers);
        }

        public void SetOptimizer(IOptimizer optimizer)
        {
            this.optimizer = optimizer;
            foreach (var layer in layers)
                layer.SetOptimizerState(optimizer);
        }

        public Matrix Forward(Matrix input, bool training = true)
        {
            Matrix current = input;
            foreach (var layer in layers)
                current = layer.Forward(current, training);
            return current;
        }

        public void Backward(Matrix gradLoss)
        {
            Matrix grad = gradLoss;
            for (int i = layers.Count - 1; i >= 0; i--)
                grad = layers[i].Backward(grad);
        }

        public void TrainStep(Matrix input, Matrix target, ILossFunction loss, double learningRate, bool clipGradients = false, double maxGradNorm = 1.0)
        {
            // Forward pass
            var predictions = Forward(input, training: true);

            // Вычисление потерь
            double lossValue = loss.Forward(predictions, target);

            // Backward pass
            var gradLoss = loss.Backward(predictions, target);
            Backward(gradLoss);

            // Градиентный спуск
            if (clipGradients)
                optimizer.ClipGradients(maxGradNorm);

            optimizer.Step(learningRate);
            optimizer.ResetGradients();
        }

        public double Evaluate(Matrix inputs, Matrix targets, ILossFunction loss)
        {
            var predictions = Forward(inputs, training: false);
            return loss.Forward(predictions, targets);
        }

        public int CountCorrect(Matrix predictions, Matrix targets)
        {
            int correct = 0;
            for (int i = 0; i < predictions.Rows; i++)
            {
                int predClass = 0;
                int targetClass = 0;
                double maxPred = predictions[i, 0];
                double maxTarget = targets[i, 0];

                for (int j = 1; j < predictions.Cols; j++)
                {
                    if (predictions[i, j] > maxPred)
                    {
                        maxPred = predictions[i, j];
                        predClass = j;
                    }
                    if (targets[i, j] > maxTarget)
                    {
                        maxTarget = targets[i, j];
                        targetClass = j;
                    }
                }
                if (predClass == targetClass)
                    correct++;
            }
            return correct;
        }
    }

    #endregion

    #region Работа с данными

    public class Dataset
    {
        public Matrix Inputs { get; }
        public Matrix Targets { get; }
        public int Size => Inputs.Rows;

        public Dataset(Matrix inputs, Matrix targets)
        {
            Inputs = inputs;
            Targets = targets;
        }

        public static Dataset FromArrays(double[][] inputs, double[][] targets)
        {
            var inputMatrix = Matrix.CreateFromRows(inputs.ToList());
            var targetMatrix = Matrix.CreateFromRows(targets.ToList());
            return new Dataset(inputMatrix, targetMatrix);
        }

        public Dataset Shuffle()
        {
            var rand = new Random();
            var indices = Enumerable.Range(0, Size).OrderBy(_ => rand.Next()).ToList();

            var newInputs = new double[Size, Inputs.Cols];
            var newTargets = new double[Size, Targets.Cols];

            for (int i = 0; i < Size; i++)
            {
                int idx = indices[i];
                for (int j = 0; j < Inputs.Cols; j++)
                    newInputs[i, j] = Inputs[idx, j];
                for (int j = 0; j < Targets.Cols; j++)
                    newTargets[i, j] = Targets[idx, j];
            }

            return new Dataset(new Matrix(newInputs), new Matrix(newTargets));
        }

        public IEnumerable<(Matrix batchInputs, Matrix batchTargets)> MiniBatches(int batchSize)
        {
            for (int i = 0; i < Size; i += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, Size - i);
                var batchInputs = new double[actualBatchSize, Inputs.Cols];
                var batchTargets = new double[actualBatchSize, Targets.Cols];

                for (int j = 0; j < actualBatchSize; j++)
                {
                    for (int k = 0; k < Inputs.Cols; k++)
                        batchInputs[j, k] = Inputs[i + j, k];
                    for (int k = 0; k < Targets.Cols; k++)
                        batchTargets[j, k] = Targets[i + j, k];
                }

                yield return (new Matrix(batchInputs), new Matrix(batchTargets));
            }
        }

        public Dataset Map(Func<double[], double[]> inputTransform, Func<double[], double[]> targetTransform = null)
        {
            var newInputs = new double[Size, Inputs.Cols];
            var newTargets = new double[Size, Targets.Cols];

            for (int i = 0; i < Size; i++)
            {
                var inputRow = new double[Inputs.Cols];
                for (int j = 0; j < Inputs.Cols; j++)
                    inputRow[j] = Inputs[i, j];

                var transformedInput = inputTransform(inputRow);
                for (int j = 0; j < Inputs.Cols; j++)
                    newInputs[i, j] = transformedInput[j];

                if (targetTransform != null)
                {
                    var targetRow = new double[Targets.Cols];
                    for (int j = 0; j < Targets.Cols; j++)
                        targetRow[j] = Targets[i, j];

                    var transformedTarget = targetTransform(targetRow);
                    for (int j = 0; j < Targets.Cols; j++)
                        newTargets[i, j] = transformedTarget[j];
                }
                else
                {
                    for (int j = 0; j < Targets.Cols; j++)
                        newTargets[i, j] = Targets[i, j];
                }
            }

            return new Dataset(new Matrix(newInputs), new Matrix(newTargets));
        }

        public Dataset Normalize()
        {
            var means = new double[Inputs.Cols];
            var stds = new double[Inputs.Cols];

            // Вычисление среднего
            for (int j = 0; j < Inputs.Cols; j++)
            {
                for (int i = 0; i < Size; i++)
                    means[j] += Inputs[i, j];
                means[j] /= Size;
            }

            // Вычисление стандартного отклонения
            for (int j = 0; j < Inputs.Cols; j++)
            {
                for (int i = 0; i < Size; i++)
                    stds[j] += (Inputs[i, j] - means[j]) * (Inputs[i, j] - means[j]);
                stds[j] = Math.Sqrt(stds[j] / Size);
                if (stds[j] < 1e-8) stds[j] = 1;
            }

            return Map(x =>
            {
                var result = new double[x.Length];
                for (int i = 0; i < x.Length; i++)
                    result[i] = (x[i] - means[i]) / stds[i];
                return result;
            });
        }

        public (Dataset train, Dataset test) Split(double testRatio = 0.2)
        {
            var shuffled = Shuffle();
            int testSize = (int)(Size * testRatio);
            int trainSize = Size - testSize;

            var trainInputs = new double[trainSize, Inputs.Cols];
            var trainTargets = new double[trainSize, Targets.Cols];
            var testInputs = new double[testSize, Inputs.Cols];
            var testTargets = new double[testSize, Targets.Cols];

            for (int i = 0; i < trainSize; i++)
            {
                for (int j = 0; j < Inputs.Cols; j++)
                    trainInputs[i, j] = shuffled.Inputs[i, j];
                for (int j = 0; j < Targets.Cols; j++)
                    trainTargets[i, j] = shuffled.Targets[i, j];
            }

            for (int i = 0; i < testSize; i++)
            {
                for (int j = 0; j < Inputs.Cols; j++)
                    testInputs[i, j] = shuffled.Inputs[trainSize + i, j];
                for (int j = 0; j < Targets.Cols; j++)
                    testTargets[i, j] = shuffled.Targets[trainSize + i, j];
            }

            return (new Dataset(new Matrix(trainInputs), new Matrix(trainTargets)),
                    new Dataset(new Matrix(testInputs), new Matrix(testTargets)));
        }
    }

    #endregion

    #region Тренировочный цикл

    public class Trainer
    {
        private readonly NeuralNetwork network;
        private readonly ILossFunction loss;
        private readonly IOptimizer optimizer;
        private double learningRate;
        private int batchSize;
        private int epochs;
        private bool clipGradients;
        private double maxGradNorm;
        private bool verbose;

        public Trainer(NeuralNetwork network, ILossFunction loss, IOptimizer optimizer)
        {
            this.network = network;
            this.loss = loss;
            this.optimizer = optimizer;
            network.SetOptimizer(optimizer);
        }

        public Trainer SetLearningRate(double lr)
        {
            learningRate = lr;
            return this;
        }

        public Trainer SetBatchSize(int size)
        {
            batchSize = size;
            return this;
        }

        public Trainer SetEpochs(int epochs)
        {
            this.epochs = epochs;
            return this;
        }

        public Trainer EnableGradientClipping(double maxNorm = 1.0)
        {
            clipGradients = true;
            maxGradNorm = maxNorm;
            return this;
        }

        public Trainer SetVerbose(bool verbose = true)
        {
            this.verbose = verbose;
            return this;
        }

        public void Fit(Dataset trainData, Dataset validationData = null)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                var shuffled = trainData.Shuffle();
                double totalLoss = 0;
                int batches = 0;

                foreach (var (batchInputs, batchTargets) in shuffled.MiniBatches(batchSize))
                {
                    network.TrainStep(batchInputs, batchTargets, loss, learningRate, clipGradients, maxGradNorm);
                    var preds = network.Forward(batchInputs, false);
                    totalLoss += loss.Forward(preds, batchTargets);
                    batches++;
                }

                double avgLoss = totalLoss / batches;

                if (verbose && (epoch % 10 == 0 || epoch == epochs - 1))
                {
                    Console.WriteLine($"Epoch {epoch + 1}/{epochs} - Loss: {avgLoss:F6}");

                    if (validationData != null)
                    {
                        var valLoss = network.Evaluate(validationData.Inputs, validationData.Targets, loss);
                        int correct = network.CountCorrect(network.Forward(validationData.Inputs, false), validationData.Targets);
                        double accuracy = 100.0 * correct / validationData.Size;
                        Console.WriteLine($"  Validation Loss: {valLoss:F6}, Accuracy: {accuracy:F2}%");
                    }
                }
            }
        }
    }

    #endregion

    #region Примеры использования

    public static class Examples
    {
        /// <summary>
        /// Пример 1: Классификация Iris
        /// </summary>
        public static void RunIrisExample()
        {
            Console.WriteLine("\n=== Пример 1: Классификация Iris ===\n");

            // Данные Iris (упрощённая версия)
            var inputs = new double[][]
            {
                new double[] {5.1, 3.5, 1.4, 0.2}, new double[] {4.9, 3.0, 1.4, 0.2},
                new double[] {4.7, 3.2, 1.3, 0.2}, new double[] {4.6, 3.1, 1.5, 0.2},
                new double[] {5.0, 3.6, 1.4, 0.2}, new double[] {5.4, 3.9, 1.7, 0.4},
                new double[] {4.6, 3.4, 1.4, 0.3}, new double[] {5.0, 3.4, 1.5, 0.2},
                new double[] {4.4, 2.9, 1.4, 0.2}, new double[] {4.9, 3.1, 1.5, 0.1},
                new double[] {7.0, 3.2, 4.7, 1.4}, new double[] {6.4, 3.2, 4.5, 1.5},
                new double[] {6.9, 3.1, 4.9, 1.5}, new double[] {5.5, 2.3, 4.0, 1.3},
                new double[] {6.5, 2.8, 4.6, 1.5}, new double[] {5.7, 2.8, 4.5, 1.3},
                new double[] {6.3, 3.3, 4.7, 1.6}, new double[] {4.9, 2.4, 3.3, 1.0},
                new double[] {6.6, 2.9, 4.6, 1.3}, new double[] {5.2, 2.7, 3.9, 1.4},
                new double[] {6.3, 3.3, 6.0, 2.5}, new double[] {5.8, 2.7, 5.1, 1.9},
                new double[] {7.1, 3.0, 5.9, 2.1}, new double[] {6.3, 2.9, 5.6, 1.8},
                new double[] {6.5, 3.0, 5.8, 2.2}, new double[] {7.6, 3.0, 6.6, 2.1},
                new double[] {4.9, 2.5, 4.5, 1.7}, new double[] {7.3, 2.9, 6.3, 1.8},
                new double[] {6.7, 2.5, 5.8, 1.8}, new double[] {7.2, 3.6, 6.1, 2.5}
            };

            // One-hot encoding целевых классов
            var targets = new double[][]
            {
                new double[] {1, 0, 0}, new double[] {1, 0, 0}, new double[] {1, 0, 0},
                new double[] {1, 0, 0}, new double[] {1, 0, 0}, new double[] {1, 0, 0},
                new double[] {1, 0, 0}, new double[] {1, 0, 0}, new double[] {1, 0, 0},
                new double[] {1, 0, 0}, new double[] {0, 1, 0}, new double[] {0, 1, 0},
                new double[] {0, 1, 0}, new double[] {0, 1, 0}, new double[] {0, 1, 0},
                new double[] {0, 1, 0}, new double[] {0, 1, 0}, new double[] {0, 1, 0},
                new double[] {0, 1, 0}, new double[] {0, 1, 0}, new double[] {0, 0, 1},
                new double[] {0, 0, 1}, new double[] {0, 0, 1}, new double[] {0, 0, 1},
                new double[] {0, 0, 1}, new double[] {0, 0, 1}, new double[] {0, 0, 1},
                new double[] {0, 0, 1}, new double[] {0, 0, 1}, new double[] {0, 0, 1}
            };

            var dataset = Dataset.FromArrays(inputs, targets);
            var (train, test) = dataset.Split(0.3);
            train = train.Normalize();
            test = test.Normalize();

            // Создание сети: 4 входа -> 10 нейронов (ReLU) -> 3 выхода (Softmax)
            var network = new NeuralNetwork(
                new DenseLayer(4, 10, new ReLU()),
                new DenseLayer(10, 3, new Softmax())
            );

            // Обучение в несколько строк
            new Trainer(network, new CrossEntropyLoss(), new Adam())
                .SetLearningRate(0.01)
                .SetBatchSize(8)
                .SetEpochs(200)
                .SetVerbose(true)
                .Fit(train, test);

            // Оценка на тесте
            var predictions = network.Forward(test.Inputs, false);
            int correct = network.CountCorrect(predictions, test.Targets);
            Console.WriteLine($"\nТестовая точность: {100.0 * correct / test.Size:F2}%");
        }

        /// <summary>
        /// Пример 2: Регрессия (аппроксимация функции)
        /// </summary>
        public static void RunRegressionExample()
        {
            Console.WriteLine("\n=== Пример 2: Регрессия ===\n");

            // Генерация данных: y = sin(x) + шум
            var rand = new Random(42);
            var inputs = new List<double[]>();
            var targets = new List<double[]>();

            for (int i = 0; i < 100; i++)
            {
                double x = i * 0.1;
                double y = Math.Sin(x) + (rand.NextDouble() - 0.5) * 0.1;
                inputs.Add(new double[] { x });
                targets.Add(new double[] { y });
            }

            var dataset = Dataset.FromArrays(inputs.ToArray(), targets.ToArray());
            var (train, test) = dataset.Split(0.2);

            // Сеть: 1 вход -> 20 нейронов (ReLU) -> 10 нейронов (ReLU) -> 1 выход (Linear)
            var network = new NeuralNetwork(
                new DenseLayer(1, 20, new ReLU()),
                new DenseLayer(20, 10, new ReLU()),
                new DenseLayer(10, 1, new Linear())
            );

            new Trainer(network, new MSELoss(), new MomentumSGD(0.9))
                .SetLearningRate(0.001)
                .SetBatchSize(16)
                .SetEpochs(500)
                .EnableGradientClipping(0.5)
                .SetVerbose(true)
                .Fit(train, test);

            // Предсказания
            Console.WriteLine("\nПример предсказаний:");
            var testPreds = network.Forward(test.Inputs, false);
            for (int i = 0; i < Math.Min(5, test.Size); i++)
            {
                Console.WriteLine($"  x={test.Inputs[i, 0]:F2}, true={test.Targets[i, 0]:F4}, pred={testPreds[i, 0]:F4}");
            }
        }

        /// <summary>
        /// Пример 3: XOR проблема
        /// </summary>
        public static void RunXorExample()
        {
            Console.WriteLine("\n=== Пример 3: XOR проблема ===\n");

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

            var dataset = Dataset.FromArrays(inputs, targets);

            // Сеть: 2 входа -> 4 нейрона (ReLU) -> 2 выхода (Softmax)
            var network = new NeuralNetwork(
                new DenseLayer(2, 4, new ReLU()),
                new DenseLayer(4, 2, new Softmax())
            );

            new Trainer(network, new CrossEntropyLoss(), new SGD())
                .SetLearningRate(0.1)
                .SetBatchSize(2)
                .SetEpochs(1000)
                .SetVerbose(false)
                .Fit(dataset);

            // Проверка результатов
            Console.WriteLine("Результаты обучения XOR:");
            var predictions = network.Forward(dataset.Inputs, false);
            for (int i = 0; i < 4; i++)
            {
                Console.WriteLine($"  [{dataset.Inputs[i, 0]}, {dataset.Inputs[i, 1]}] -> " +
                    $"[{predictions[i, 0]:F3}, {predictions[i, 1]:F3}] -> " +
                    $"{(predictions[i, 0] > predictions[i, 1] ? 0 : 1)}");
            }
        }
    }

    #endregion

    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("╔═══════════════════════════════════════════════════════════╗");
            Console.WriteLine("║     Нейросетевой фреймворк на C#                          ║");
            Console.WriteLine("║     Полносвязные сети с разными оптимизаторами            ║");
            Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");

            // Запуск примеров
            Examples.RunXorExample();
            Examples.RunIrisExample();
            Examples.RunRegressionExample();

            Console.WriteLine("\n=== Все примеры завершены ===");
        }
    }
}
