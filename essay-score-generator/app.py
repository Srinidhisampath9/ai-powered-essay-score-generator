from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import textstat
import io
import base64
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

app = Flask(__name__)

# ✅ Load Pretrained Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ✅ Define Bi-LSTM Model for Grammar Checking
class BiLSTMGrammarModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(BiLSTMGrammarModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out[:, -1, :])  # Taking output from last time step
        return logits

# ✅ Load Pretrained Bi-LSTM Model (Replace with Your Own Model)
vocab_size = tokenizer.vocab_size
embedding_dim = 128
hidden_dim = 256
output_dim = 1

# Dummy Model (Replace with a Trained Model)
model = BiLSTMGrammarModel(vocab_size, embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("bilstm_model.pth", map_location=torch.device('cpu')))  # Load trained model
model.eval()

def evaluate_grammar(essay_text):
    """Predicts grammar quality using Bi-LSTM."""
    tokens = tokenizer.encode(essay_text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        logits = model(tokens)
    grammar_score = torch.sigmoid(logits).item() * 10  # Normalize to 0-10 scale
    return np.clip(grammar_score, 5, 10)

def process_essay(essay_text):
    """
    Evaluates an essay using Bi-LSTM for grammar and other metrics for structure, and vocabulary.
    Returns a score out of 10.
    """

    # ✅ 1. Grammar Score (Using Bi-LSTM)
    grammar_score = evaluate_grammar(essay_text)

    # ✅ 2. Clarity Score (Sentence Complexity & Readability)
    readability = textstat.flesch_reading_ease(essay_text)
    clarity_score = np.clip((readability - 30) / 7, 7, 10)

    # ✅ 3. Structure Score (Paragraph Flow)
    paragraph_count = essay_text.count("\n") + 1
    structure_score = np.clip(paragraph_count * 2, 6, 10)

    # ✅ 4. Vocabulary Score (Lexical Diversity)
    words = essay_text.split()
    unique_words_ratio = len(set(words)) / max(1, len(words))  # Avoid division by zero
    vocabulary_score = np.clip(unique_words_ratio * 10, 6, 10)

    # ✅ 5. Final Score Calculation (Balanced Weights)
    weights = {"grammar": 0.4, "clarity": 0.2, "structure": 0.2, "vocabulary": 0.2}
    attribute_scores = [grammar_score, clarity_score, structure_score, vocabulary_score]
    final_score = round(np.average(attribute_scores, weights=list(weights.values())), 1)

    attributes = {
        "Grammar": round(grammar_score, 1),
        "Clarity": round(clarity_score, 1),
        "Structure": round(structure_score, 1),
        "Vocabulary": round(vocabulary_score, 1)
    }

    # 🛠 Debugging - Print Scores for Analysis
    print(f"Grammar Score: {grammar_score}")
    print(f"Clarity Score: {clarity_score}")
    print(f"Structure Score: {structure_score}")
    print(f"Vocabulary Score: {vocabulary_score}")
    print(f"Final Score: {final_score}")

    return final_score, attributes

def generate_plot(attribute_scores, attributes):
    """ Generates a bar chart for attribute analysis. """
    plt.figure(figsize=(6, 4))
    plt.bar(attributes, attribute_scores, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel("Attributes")
    plt.ylabel("Score")
    plt.title("Essay Attribute Analysis")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        essay_text = request.form.get("essay_text", "")
        file = request.files.get("file")

        if file:
            essay_text = file.read().decode("utf-8")

        if not essay_text.strip():
            return render_template("submit.html", error="Please provide an essay to analyze.")

        score, attributes = process_essay(essay_text)
        plot_url = generate_plot(list(attributes.values()), list(attributes.keys()))

        # Verdict based on final score
        verdict = "Excellent Essay" if score > 7 else "Good Essay" if score > 5 else "Weak Essay"

        return render_template("result.html", score=round(score, 1), verdict=verdict, plot_url=plot_url)

    return render_template("submit.html")

if __name__ == '__main__':
    app.run(debug=True)
