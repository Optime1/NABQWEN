using System;

namespace NeuralFramework
{
    #region Оптимизаторы

    /// <summary>
    /// Базовый класс для оптимизаторов
    /// </summary>
    public abstract class Optimizer
    {
        protected double learningRate;
        protected double maxGradientNorm; // Для gradient clipping

        protected Optimizer(double learningRate, double maxGradientNorm = 5.0)
        {
            this.learningRate = learningRate;
            this.maxGradientNorm = maxGradientNorm;
        }

        public abstract void UpdateWeights(DenseLayer layer);
        
        /// <summary>
        /// Gradient Clipping - обрезка градиентов по норме
        /// </summary>
        protected Matrix ClipGradients(Matrix gradients)
        {
            double norm = 0;
            for (int i = 0; i < gradients.Rows; i++)
                for (int j = 0; j < gradients.Cols; j++)
                    norm += gradients[i, j] * gradients[i, j];
            norm = Math.Sqrt(norm);

            if (norm > maxGradientNorm && norm > 0)
            {
                var clipped = new Matrix(gradients.Rows, gradients.Cols);
                double scale = maxGradientNorm / norm;
                for (int i = 0; i < gradients.Rows; i++)
                    for (int j = 0; j < gradients.Cols; j++)
                        clipped[i, j] = gradients[i, j] * scale;
                return clipped;
            }
            // Возвращаем копию, чтобы не модифицировать оригинал
            var result = new Matrix(gradients.Rows, gradients.Cols);
            for (int i = 0; i < gradients.Rows; i++)
                for (int j = 0; j < gradients.Cols; j++)
                    result[i, j] = gradients[i, j];
            return result;
        }
    }

    /// <summary>
    /// SGD (Stochastic Gradient Descent)
    /// </summary>
    public class SGDOptimizer : Optimizer
    {
        public SGDOptimizer(double learningRate, double maxGradientNorm = 5.0) 
            : base(learningRate, maxGradientNorm) { }

        public override void UpdateWeights(DenseLayer layer)
        {
            // Получаем градиенты из слоя
            var weightGrad = layer.WeightGradients;
            var biasGrad = layer.BiasGradients;
            
            // Обрезка градиентов
            var clippedWeightGrad = ClipGradients(weightGrad);
            var clippedBiasGrad = ClipGradients(biasGrad);
            
            // w = w - lr * grad (обновляем веса напрямую в слое)
            for (int i = 0; i < layer.Weights.Rows; i++)
                for (int j = 0; j < layer.Weights.Cols; j++)
                    layer.Weights[i, j] -= learningRate * clippedWeightGrad[i, j];
            
            for (int i = 0; i < layer.Biases.Rows; i++)
                for (int j = 0; j < layer.Biases.Cols; j++)
                    layer.Biases[i, j] -= learningRate * clippedBiasGrad[i, j];
        }
    }

    /// <summary>
    /// Momentum SGD - SGD с инерцией
    /// </summary>
    public class MomentumSGDOptimizer : Optimizer
    {
        private readonly double momentum;
        private System.Collections.Generic.Dictionary<int, (Matrix w, Matrix b)> velocity = new();

        public MomentumSGDOptimizer(double learningRate, double momentum = 0.9, double maxGradientNorm = 5.0) 
            : base(learningRate, maxGradientNorm)
        {
            this.momentum = momentum;
        }

        public override void UpdateWeights(DenseLayer layer)
        {
            int layerId = System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(layer);
            
            if (!velocity.ContainsKey(layerId))
            {
                velocity[layerId] = (
                    Matrix.Zeros(layer.Weights.Rows, layer.Weights.Cols),
                    Matrix.Zeros(layer.Biases.Rows, layer.Biases.Cols)
                );
            }

            // Получаем градиенты из слоя
            var weightGrad = layer.WeightGradients;
            var biasGrad = layer.BiasGradients;
            
            // Обрезка градиентов
            var clippedWeightGrad = ClipGradients(weightGrad);
            var clippedBiasGrad = ClipGradients(biasGrad);

            // v = momentum * v - lr * grad
            // w = w + v
            var newVelW = new Matrix(layer.Weights.Rows, layer.Weights.Cols);
            var newVelB = new Matrix(layer.Biases.Rows, layer.Biases.Cols);
            
            for (int i = 0; i < layer.Weights.Rows; i++)
                for (int j = 0; j < layer.Weights.Cols; j++)
                {
                    newVelW[i, j] = momentum * velocity[layerId].w[i, j] - learningRate * clippedWeightGrad[i, j];
                }
            
            for (int i = 0; i < layer.Biases.Rows; i++)
                for (int j = 0; j < layer.Biases.Cols; j++)
                {
                    newVelB[i, j] = momentum * velocity[layerId].b[i, j] - learningRate * clippedBiasGrad[i, j];
                }

            velocity[layerId] = (newVelW, newVelB);

            for (int i = 0; i < layer.Weights.Rows; i++)
                for (int j = 0; j < layer.Weights.Cols; j++)
                    layer.Weights[i, j] += newVelW[i, j];
            
            for (int i = 0; i < layer.Biases.Rows; i++)
                for (int j = 0; j < layer.Biases.Cols; j++)
                    layer.Biases[i, j] += newVelB[i, j];
        }
    }

    /// <summary>
    /// Adam Optimizer - адаптивный момент
    /// </summary>
    public class AdamOptimizer : Optimizer
    {
        private readonly double beta1 = 0.9;
        private readonly double beta2 = 0.999;
        private readonly double epsilon = 1e-8;
        private int timestep = 0;
        private System.Collections.Generic.Dictionary<int, (Matrix mW, Matrix vW, Matrix mB, Matrix vB)> state = new();

        public AdamOptimizer(double learningRate = 0.001, double maxGradientNorm = 5.0) 
            : base(learningRate, maxGradientNorm) { }

        public override void UpdateWeights(DenseLayer layer)
        {
            timestep++;
            int layerId = System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(layer);

            if (!state.ContainsKey(layerId))
            {
                state[layerId] = (
                    Matrix.Zeros(layer.Weights.Rows, layer.Weights.Cols),
                    Matrix.Zeros(layer.Weights.Rows, layer.Weights.Cols),
                    Matrix.Zeros(layer.Biases.Rows, layer.Biases.Cols),
                    Matrix.Zeros(layer.Biases.Rows, layer.Biases.Cols)
                );
            }

            // Получаем градиенты из слоя
            var weightGrad = layer.WeightGradients;
            var biasGrad = layer.BiasGradients;
            
            // Обрезка градиентов
            var clippedWeightGrad = ClipGradients(weightGrad);
            var clippedBiasGrad = ClipGradients(biasGrad);

            // Обновление моментов
            for (int i = 0; i < layer.Weights.Rows; i++)
                for (int j = 0; j < layer.Weights.Cols; j++)
                {
                    state[layerId].mW[i, j] = beta1 * state[layerId].mW[i, j] + (1 - beta1) * clippedWeightGrad[i, j];
                    state[layerId].vW[i, j] = beta2 * state[layerId].vW[i, j] + (1 - beta2) * clippedWeightGrad[i, j] * clippedWeightGrad[i, j];
                    
                    var mHat = state[layerId].mW[i, j] / (1 - Math.Pow(beta1, timestep));
                    var vHat = state[layerId].vW[i, j] / (1 - Math.Pow(beta2, timestep));
                    
                    layer.Weights[i, j] -= learningRate * mHat / (Math.Sqrt(vHat) + epsilon);
                }

            for (int i = 0; i < layer.Biases.Rows; i++)
                for (int j = 0; j < layer.Biases.Cols; j++)
                {
                    state[layerId].mB[i, j] = beta1 * state[layerId].mB[i, j] + (1 - beta1) * clippedBiasGrad[i, j];
                    state[layerId].vB[i, j] = beta2 * state[layerId].vB[i, j] + (1 - beta2) * clippedBiasGrad[i, j] * clippedBiasGrad[i, j];
                    
                    var mHat = state[layerId].mB[i, j] / (1 - Math.Pow(beta1, timestep));
                    var vHat = state[layerId].vB[i, j] / (1 - Math.Pow(beta2, timestep));
                    
                    layer.Biases[i, j] -= learningRate * mHat / (Math.Sqrt(vHat) + epsilon);
                }
        }
    }

    #endregion
}
