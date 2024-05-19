namespace MLSentimentAnalysis.Models
{
    public class Chat
    {
        public class ChatResponse
        {
            public string id { get; set; }
            public string created { get; set; }
            public string model { get; set; }
            public string[] choices { get; set; }
        }

        public class ChatRequestModel
        {
            public string apiKey { get; set; }
            public string prompt { get; set; }
        }
    }
}
