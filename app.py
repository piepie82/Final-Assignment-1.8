from flask import Flask,render_template,request
import google.generativeai as palm
from textblob import TextBlob
from transformers import pipeline

api = "AIzaSyBrSRQOusWyAMBfm1F27g6Ci97xk5-J258"
palm.configure(api_key=api)
model = {"model": "models/text-bison-001"}

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    return(render_template("index.html"))

@app.route("/financial_FAQ",methods=["GET","POST"])
def financial_FAQ():
    return(render_template("financial_FAQ.html"))

@app.route("/makersuite",methods=["GET","POST"])
def makersuite():
    q = request.form.get("q")
    r = palm.generate_text(prompt=q, **model)
    return(render_template("makersuite.html",r=r.result))

@app.route("/joke", methods=["GET","POST"])
def joke():
    joke_text = "Which noodle is the heaviest? Wanton (one-tonne) noodles."
    return render_template("joke.html", joke=joke_text)

@app.route("/textblob_sentiment", methods=["GET", "POST"])
def textblob_sentiment():
    sentiment = None
    if request.method == "POST":
        user_text = request.form.get("user_text")
        sentiment = TextBlob(user_text).sentiment
    return render_template("textblob_sentiment.html", sentiment=sentiment)

@app.route("/transformer_sentiment", methods=["GET", "POST"])
def transformer_sentiment():
    sentiment = None
    if request.method == "POST":
        user_text = request.form.get("user_text")
        try:
            transformer_model = pipeline("sentiment-analysis")
            sentiment = transformer_model(user_text)[0]
        except Exception as e:
            sentiment = {"label": "Error", "score": 0.0, "message": str(e)}
    return render_template("transformer_sentiment.html", sentiment=sentiment)

@app.template_filter('format_sentiment')
def format_sentiment(sentiment):
    """Formats sentiment polarity into a human-readable string"""
    if isinstance(sentiment, dict):
        if sentiment.get("label") == "POSITIVE":
            return "Positive"
        elif sentiment.get("label") == "NEGATIVE":
            return "Negative"
        else:
            return "Neutral"
    elif hasattr(sentiment, 'polarity'):
        if sentiment.polarity > 0:
            return "Positive"
        elif sentiment.polarity < 0:
            return "Negative"
        else:
            return "Neutral"
    return "Unknown"


if __name__ == "__main__":
    app.run()
