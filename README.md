# robust-sentiment-swedish-api-limited

This repository provides a **FastAPI-based** sentiment analysis API using **KBLab's Swedish BERT models**. The API consists of two microservices:
- **Tokenization Service**: Converts raw text into tokenized format.
- **Sentiment Analysis Service**: Classifies sentiment based on tokenized input.

## **Features**
* Uses **KBLab's Swedish BERT models** for tokenization and sentiment classification.
* Supports **long texts** with **512-token sliding windows (50% overlap)**.
* **Modular architecture** with separated routes for tokenization and sentiment analysis.
* Fully **containerisable** (can be deployed with Docker or cloud services).
* FastAPI framework ensures **scalability and performance**.

---

## **Installation**

### **Clone the Repository**
```sh
 git clone https://github.com/TeaElming/robust-sentiment-swedish-api-limited.git
 cd robust-sentiment-swedish-api-limited
```

### **Set Up a Virtual Environment**
```sh
python -m venv sentimentAPI
source sentimentAPI/bin/activate  # Mac/Linux
sentimentAPI\Scripts\activate    # Windows
```

### **Install Dependencies**
```sh
pip install -r requirements.txt
```

---

## **Configuration**
Modify `config.py` if needed:
```python
TOKENIZER_MODEL = "KBLab/megatron-bert-large-swedish-cased-165k"
SENTIMENT_MODEL = "KBLab/robust-swedish-sentiment-multiclass"

MAX_LENGTH = 512
SLIDING_WINDOW = 0.5  # 50% overlap
OVERLAP = int(MAX_LENGTH * SLIDING_WINDOW)  # 256 tokens

API_PORT = 8000
TOKENIZER_PORT = 5001
```

---

## **Running the API**

### **Start the FastAPI Server**
```sh
python main.py
```
The server will start at:
```
http://127.0.0.1:8000
```

You can access the **interactive API documentation** at:
```
http://127.0.0.1:8000/docs
```

---

## **API Usage**

### **Tokenize Text**
**Endpoint:** `POST /tokenize`
**Description:** Converts raw text into tokenized format for model input.

#### **Request Example**
```json
{
  "text": "Rickard Andersson ska ha blivit av med sitt ekonomiska bistånd eftersom han inte sökt jobb."
}
```

#### **Response Example**
```json
{
  "input_ids": [[101, 2135, 4367, ...]],
  "attention_mask": [[1, 1, 1, ...]]
}
```

### **Sentiment Analysis**
**Endpoint:** `POST /get-sentiment`
**Description:** Accepts text input, tokenizes it, and returns sentiment classification.

#### **Request Example**
```json
{
  "input_ids": [[101, 2135, 4367, ...]],
  "attention_mask": [[1, 1, 1, ...]]
}
```

#### **Response Example**
```json
{
  "label": "NEGATIVE",
  "score": 0.87
}
```

---

## **Project Structure**
```
/backend
│── /models
│   ├── sentiment_model.py      # Sentiment analysis logic
│   ├── tokenizer_model.py      # Text tokenization logic
│── /routes
│   ├── sentiment_route.py      # API route for sentiment analysis
│   ├── tokenizer_route.py      # API route for tokenization
│── main.py                     # FastAPI app entry point
│── config.py                    # Configuration settings
│── requirements.txt              # Python dependencies
```

---

## **Running with Docker (Optional)**

If you prefer to run this API as a containerized service:

### **Build the Docker Image**
```sh
docker build -t sentiment-api .
```

### **Run the Docker Container**
```sh
docker run -p 8000:8000 sentiment-api
```

The API will now be available at `http://127.0.0.1:8000/docs`

---

## **Next Steps**
- 🔹 **Optimise API performance** by using GPU (if available).
- 🔹 **Deploy API** to cloud (AWS, Azure, GCP, etc.).
- 🔹 **Improve response format** for better insights.

