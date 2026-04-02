using System;

namespace NeuralFramework
{
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

        public static Matrix CreateFromRows(double[][] rows)
        {
            int r = rows.Length;
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

        public static Matrix operator *(double scalar, Matrix a) => a * scalar;

        public Matrix Transpose()
        {
            var result = new Matrix(Cols, Rows);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result.Data[j, i] = Data[i, j];
            return result;
        }

        public Matrix AddBias()
        {
            var result = new Matrix(Rows, Cols + 1);
            for (int i = 0; i < Rows; i++)
            {
                result.Data[i, 0] = 1.0;
                for (int j = 0; j < Cols; j++)
                    result.Data[i, j + 1] = Data[i, j];
            }
            return result;
        }

        public Matrix RemoveBias()
        {
            if (Cols < 2) return Copy();
            var result = new Matrix(Rows, Cols - 1);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols - 1; j++)
                    result.Data[i, j] = Data[i, j + 1];
            return result;
        }

        public double[] Row(int index)
        {
            var row = new double[Cols];
            for (int j = 0; j < Cols; j++)
                row[j] = Data[index, j];
            return row;
        }

        public void SetRow(int index, double[] values)
        {
            for (int j = 0; j < Cols; j++)
                Data[index, j] = values[j];
        }

        public double Sum()
        {
            double sum = 0;
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    sum += Data[i, j];
            return sum;
        }

        public double Mean() => Sum() / (Rows * Cols);

        public Matrix Apply(Func<double, double> func)
        {
            var result = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result.Data[i, j] = func(Data[i, j]);
            return result;
        }

        public void InPlaceApply(Func<double, double> func)
        {
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Data[i, j] = func(Data[i, j]);
        }

        public override string ToString()
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"Matrix[{Rows}x{Cols}]:");
            for (int i = 0; i < Math.Min(Rows, 10); i++)
            {
                sb.Append("  ");
                for (int j = 0; j < Math.Min(Cols, 10); j++)
                    sb.Append($"{Data[i, j],8:F4} ");
                if (Cols > 10) sb.Append("...");
                sb.AppendLine();
            }
            if (Rows > 10) sb.AppendLine("  ...");
            return sb.ToString();
        }
    }
}
