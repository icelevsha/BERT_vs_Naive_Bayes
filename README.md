# 📝 Sentiment Analysis: BERT (RoBERTa-base) vs Naive Bayes

В этом проекте я провёл сравнение двух подходов к анализу тональности текста:

- 🔹 **Наивный Байес** — простая классическая модель для NLP.  
- 🔹 **Дообученный BERT (RoBERTa-base)** — современная трансформерная модель, дообученная на кастомном [датасете](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).  

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
model_name = "levshaone/model_sentiment"
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
```
### 🔹 Интеграция в Telegram-бота
После обучения и загрузки модели на Hugging Face её можно сразу применять в реальных проектах.  
Один из удобных способов — интеграция в **Telegram-бота**, который будет анализировать сообщения пользователей в режиме реального времени.
Ниже представлена реализация бота с интеграцией модели
```python
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "levshaone/model_sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

labels = {
    0: "Negative",
    1: "Positive"
}

def analyze_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    label_id = torch.argmax(probs, dim=-1).item()
    label_name = labels[label_id]
    return f"Sentiment: {label_name}"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    result = analyze_text(text)
    await update.message.reply_text(result)

if __name__ == "__main__":
    TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"  # токен бота
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot started...")
    app.run_polling()
```

## 📱 Примеры работы Telegram-бота

Ниже показаны примеры предсказаний модели прямо в чате:

![positive example](https://github.com/icelevsha/BERT_vs_Naive_Bayes/blob/main/images/negative.png)


### 🔹 Негативное сообщение
![negative example](https://github.com/icelevsha/BERT_vs_Naive_Bayes/blob/main/images/positive.png)

## 🔜 Планы на будущее
- Добавить поддержку русского языка через дообучение модели на русскоязычных текстах.
- Улучшить точность анализа для длинных сообщений.
