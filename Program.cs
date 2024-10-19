using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SalaryPrediction
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            string dataPath = "employee_salary.csv";  
            IDataView dataView = mlContext.Data.LoadFromTextFile<SalaryData>(dataPath, hasHeader: true, separatorChar: ',');

            var dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.1);
            var trainData = dataSplit.TrainSet;
            var testData = dataSplit.TestSet;

            // Define the pipeline
            var pipeline = mlContext.Transforms.CopyColumns("Label", "Salary") 
                .Append(mlContext.Transforms.Concatenate("Features", "YearsExperience")) 
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features")); 

            var model = pipeline.Fit(trainData);

            var testPredictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(testPredictions, labelColumnName: "Label");

            Console.WriteLine($"R^2: {metrics.RSquared}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");

            var predictionEngine = mlContext.Model.CreatePredictionEngine<SalaryData, SalaryPrediction>(model);

            var input = new SalaryData { YearsExperience = 1 };
            var prediction = predictionEngine.Predict(input);
            Console.WriteLine($"Predicted salary for {input.YearsExperience} years of experience: {prediction.Score}");
        }
    }

    public class SalaryData
    {
        [LoadColumn(0)]
        public float YearsExperience { get; set; }

        [LoadColumn(1)]
        public float Salary { get; set; }
    }

    public class SalaryPrediction
    {
        public float Score { get; set; }  
    }
}
