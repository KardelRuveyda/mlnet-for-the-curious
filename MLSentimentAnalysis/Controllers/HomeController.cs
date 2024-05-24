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
        private readonly string _yelpDataPath = Path.Combine(Directory.GetCurrentDirectory(), "Data", "yelp_labelled.txt");
        private readonly MLContext _mlContext = new MLContext();
        private ITransformer _model;
        private readonly IHttpClientFactory _clientFactory;


        public HomeController(IHttpClientFactory clientFactory)
        {
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
                // Veri k�mesini y�kle
                TrainTestData splitDataView = LoadData(_mlContext);

                // Modeli e�it
                _model = BuildAndTrainModel(_mlContext, splitDataView.TrainSet);

                // Modeli de�erlendir
                Evaluate(_mlContext, _model, splitDataView.TestSet);

                // E�itim sonu�lar�n� ekrana yazd�r
                ViewBag.TrainResult = "Model ba�ar�yla E�itildi!";
                ViewBag.Accuracy = ViewBag.Accuracy ?? 0.0; // E�er accuracy de�eri null ise, 0.0 olarak ata
                ViewBag.Auc = ViewBag.Auc ?? 0.0; // E�er auc de�eri null ise, 0.0 olarak ata
                ViewBag.F1Score = ViewBag.F1Score ?? 0.0; // E�er f1Score de�eri null ise, 0.0 olarak ata
                ViewBag.Train = true;
            }
            catch (Exception ex)
            {
                ViewBag.TrainResult = "E�itim s�ras�nda bir hata olu�tu: " + ex.Message;
            }

            // Index view'�n� �a��r
            return View("TrainModelView");
        }


        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
  
        private TrainTestData LoadData(MLContext mLContext)
        {
            // 1. Ad�m: Veri K�mesini Y�kleme
            IDataView dataView = _mlContext.Data.LoadFromTextFile<SentimentData>(_yelpDataPath, hasHeader: false);

            // 2. Ad�m: Veri K�mesini E�itim ve Test Olarak Ay�rma
            TrainTestData splitDataView = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // 3. Ad�m: E�itim ve Test Verilerini Geri D�nd�rme
            return splitDataView;
        }

        private ITransformer BuildAndTrainModel(MLContext mLContext, IDataView splitTrainSet)
        {
            // 1. Ad�m: Estimator Olu�turma
            var estimator = _mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            // 2. Ad�m: Modeli E�itme
            var model = estimator.Fit(splitTrainSet);

            // 3. Ad�m: E�itilmi� Modeli D�nd�rme
            return model;
        }

        private SentimentPrediction GetPredictionForReviewContent(MLContext mlContext, ITransformer model, string reviewContent)
        {
            // 1. Ad�m: Prediction Engine Olu�turma
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            // 2. Ad�m: �rnek Veri Olu�turma
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = reviewContent
            };

            // 3. Ad�m: Tahmin Yapma
            var resultPrediction = predictionFunction.Predict(sampleStatement);

            // 4. Ad�m: Tahmin Sonucunu D�nd�rme
            return resultPrediction;
        }

        private void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            // 1. Ad�m: Tahminler Yapma
            IDataView predictions = model.Transform(splitTestSet);

            // 2. Ad�m: Modeli De�erlendirme
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            // 3. Ad�m: Performans Metriklerini ViewBag'e Atama
            ViewBag.Accuracy = metrics.Accuracy;
            ViewBag.Auc = metrics.AreaUnderRocCurve;
            ViewBag.F1Score = metrics.F1Score;
        }


        [HttpPost]
        public IActionResult PredictSentiment(string reviewContent)
        {
            //Modeli e�it
            TrainTestData splitDataView = LoadData(_mlContext);
            _model = BuildAndTrainModel(_mlContext, splitDataView.TrainSet);

            var resultPrediction = GetPredictionForReviewContent(_mlContext, _model, reviewContent);

            return Json(resultPrediction);
        }

        [HttpPost]
        public async Task<IActionResult> SendChatGpt(ChatRequestModel model)
        {
            // 1. Ad�m: Girdi Kontrol�
            if (string.IsNullOrWhiteSpace(model.apiKey) || string.IsNullOrWhiteSpace(model.prompt))
            {
                return BadRequest("API key and evaluation are required.");
            }

            // 2. Ad�m: De�i�kenlerin Tan�mlanmas�
            string outputResult = "";
            string apiUrl = "https://api.openai.com/v1/chat/completions";

            // 3. Ad�m: �stek G�vdesinin Haz�rlanmas�
            var requestBody = new
            {
                model = "gpt-4",
                messages = new[]
        {
            new { role = "system", content = "Sen yard�mc� bir asistans�n." },
            new { role = "user", content = model.prompt }
        },
                max_tokens = 300
            };

            // 4. Ad�m: API �ste�i G�nderme
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

