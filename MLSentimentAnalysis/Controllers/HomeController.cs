using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLSentimentAnalysis.Models;
using OpenAI_API.Completions;
using OpenAI_API;
using System.Diagnostics;
using System.Text.Json;
using static Microsoft.ML.DataOperationsCatalog;
using static MLSentimentAnalysis.Models.Chat;
using OpenAI_API.Chat;
using Newtonsoft.Json;
using System.Net.Http;
using System.Text;

namespace MLSentimentAnalysis.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly string _yelpDataPath = Path.Combine(Directory.GetCurrentDirectory(), "Data", "yelp_labelled.txt");
        private readonly MLContext _mlContext = new MLContext();
        private ITransformer _model;
        private readonly IHttpClientFactory _clientFactory;


        public HomeController(ILogger<HomeController> logger, IHttpClientFactory clientFactory)
        {
            _logger = logger;
            _clientFactory = clientFactory;
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult TrainModelView()
        {
            return View();
        }

        public IActionResult TrainModel()
        {
            try
            {
                // Veri kümesini yükle
                TrainTestData splitDataView = LoadData(_mlContext);

                // Modeli eðit
                _model = BuildAndTrainModel(_mlContext, splitDataView.TrainSet);

                // Modeli deðerlendir
                Evaluate(_mlContext, _model, splitDataView.TestSet);

                // Eðitim sonuçlarýný ekrana yazdýr
                ViewBag.TrainResult = "Model baþarýyla Eðitildi!";
                ViewBag.Accuracy = ViewBag.Accuracy ?? 0.0; // Eðer accuracy deðeri null ise, 0.0 olarak ata
                ViewBag.Auc = ViewBag.Auc ?? 0.0; // Eðer auc deðeri null ise, 0.0 olarak ata
                ViewBag.F1Score = ViewBag.F1Score ?? 0.0; // Eðer f1Score deðeri null ise, 0.0 olarak ata
                ViewBag.Train = true;
            }
            catch (Exception ex)
            {
                ViewBag.TrainResult = "Eðitim sýrasýnda bir hata oluþtu: " + ex.Message;
            }

            // Index view'ýný çaðýr
            return View("TrainModelView");
        }


        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
  
        private TrainTestData LoadData(MLContext mLContext)
        {
            IDataView dataView = _mlContext.Data.LoadFromTextFile<SentimentData>(_yelpDataPath, hasHeader: false);
            TrainTestData splitDataView = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }

        private ITransformer BuildAndTrainModel(MLContext mLContext, IDataView splitTrainSet)
        {
            var estimator = _mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            var model = estimator.Fit(splitTrainSet);
            return model;
        }

        private SentimentPrediction GetPredictionForReviewContent(MLContext mlContext, ITransformer model, string reviewContent)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = reviewContent
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);
            return resultPrediction;
        }

        private void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            ViewBag.Accuracy = metrics.Accuracy;
            ViewBag.Auc = metrics.AreaUnderRocCurve;
            ViewBag.F1Score = metrics.F1Score;
        }


        [HttpPost]
        public IActionResult PredictSentiment(string reviewContent)
        {
            //Modeli eðit
            TrainTestData splitDataView = LoadData(_mlContext);
            _model = BuildAndTrainModel(_mlContext, splitDataView.TrainSet);

            if (_model == null)
            {
                ViewBag.ErrorMessage = "Model is not trained yet. Please train the model first.";
                return View("TrainModelView");
            }

            var resultPrediction = GetPredictionForReviewContent(_mlContext, _model, reviewContent);

            return Json(resultPrediction);
        }

        [HttpPost]
        public async Task<IActionResult> SendChatGpt(ChatRequestModel model)
        {
            if (string.IsNullOrWhiteSpace(model.apiKey) || string.IsNullOrWhiteSpace(model.prompt))
            {
                return BadRequest("API key and evaluation are required.");
            }

            string outputResult = "";
            string apiUrl = "https://api.openai.com/v1/chat/completions";

            var requestBody = new
            {
                model = "gpt-4",
                messages = new[]
        {
            new { role = "system", content = "Sen yardýmcý bir asistansýn." },
            new { role = "user", content = model.prompt }
        },
                max_tokens = 300
            };

            var requestContent = new StringContent(JsonConvert.SerializeObject(requestBody), Encoding.UTF8, "application/json");

            try
            {
                var client = _clientFactory.CreateClient();
                client.DefaultRequestHeaders.Add("Authorization", $"Bearer {model.apiKey}");

                var response = await client.PostAsync(apiUrl, requestContent);
                response.EnsureSuccessStatusCode();

                var responseBody = await response.Content.ReadAsStringAsync();
                var responseObject = JsonConvert.DeserializeObject<dynamic>(responseBody);

                foreach (var choice in responseObject.choices)
                {
                    outputResult += choice.message.content;
                }

                return Ok(outputResult);
            }
            catch (HttpRequestException ex)
            {
                return StatusCode(500, $"Internal server error: {ex.Message}");
            }
        }

    }

}

