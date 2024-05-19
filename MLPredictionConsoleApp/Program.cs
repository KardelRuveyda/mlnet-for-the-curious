using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MyMLApp
{
    public class Program
    {
        public class HouseData
        {
            public float Size { get; set; }
            public float Price { get; set; }
        }

        public class Prediction
        {
            [ColumnName("Score")]
            public float Price { get; set; }
        }

        static void Main(string[] args)
        {
            // Create a new ML context, for ML.NET operations. It is a centralized context for all ML.NET operations.
            var context = new MLContext();

            // Create some example data
            var houseData = new[]
            {
                new HouseData { Size = 1.1F, Price = 1.2F },
                new HouseData { Size = 1.9F, Price = 2.3F },
                new HouseData { Size = 2.8F, Price = 3.0F },
                new HouseData { Size = 3.4F, Price = 3.7F }
            };

            // Convert the data into an IDataView
            var data = context.Data.LoadFromEnumerable(houseData);

            // Create a learning pipeline
            var pipeline = context.Transforms.Concatenate("Features", "Size")
                .Append(context.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

            // Train the model
            var model = pipeline.Fit(data);

            // Create a prediction engine from the model
            var size = new HouseData { Size = 2.5F };
            var predictionEngine = context.Model.CreatePredictionEngine<HouseData, Prediction>(model);

            // Use the model to predict the price of the house
            var price = predictionEngine.Predict(size);
            Console.WriteLine($"Predicted price for house with size: {size.Size} is: {price.Price}");
        }
    }
}
