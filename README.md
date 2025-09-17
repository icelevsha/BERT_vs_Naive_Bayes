# 📝 Sentiment Analysis: BERT (RoBERTa-base) vs Naive Bayes

В этом проекте я провёл сравнение двух подходов к анализу тональности текста:

- 🔹 **Наивный Байес** — простая классическая модель для NLP.  
- 🔹 **Дообученный BERT (RoBERTa-base)** — современная трансформерная модель, дообученная на кастомном датасете.  

---

## 📊 Результаты

- **Наивный Байес** справляется с короткими текстами, но хуже работает на длинных и сложных примерах.  
- **RoBERTa-base (fine-tuned)** показывает значительно более высокую точность и устойчивость к вариативным формулировкам.  

Финальная модель доступна на **Hugging Face Hub**:  
👉 [levshaone/model_sentiment](https://huggingface.co/levshaone/model_sentiment)

---

## 🚀 Использование модели

### 🔹 Прямое предсказание
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# загрузка модели из Hugging Face Hub
model_name = "levshaone/sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

id2label = {0: "negative", 1: "positive"}

def predict(text: str):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**encoding)
        preds = torch.argmax(outputs.logits, dim=-1).item()
    return id2label[preds]

print(predict("I really love this movie!"))   # positive
print(predict("This film was terrible..."))   # negative

### 🔹 Интеграция в Telegram-бота
После обучения и загрузки модели на Hugging Face её можно сразу применять в реальных проектах.  
Один из удобных способов — интеграция в **Telegram-бота**, который будет анализировать сообщения пользователей в режиме реального времени.

(тут твой код бота)

---

## 📱 Примеры работы Telegram-бота

Ниже показаны примеры предсказаний модели прямо в чате:

### 🔹 Позитивное сообщение
![positive example](https://github.com/icelevsha/BERT_vs_Naive_Bayes/blob/main/images/negative.png)


### 🔹 Негативное сообщение
![negative example](images/negative_example.png)
