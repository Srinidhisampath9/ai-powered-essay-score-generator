# ai-powered-essay-score-generator

📌 Project Overview

The AI-Enhanced Smart Essay Assessment and Scoring System is a web-based application built using Flask that evaluates essays based on multiple attributes such as grammar, clarity, structure, and vocabulary. The system provides a score out of 10 and generates visualizations to highlight strengths and weaknesses in the essay.

🚀 Features

AI-Based Essay Evaluation using NLP techniques.

Scoring Metrics: Grammar, Clarity, Structure, and Vocabulary.

Visual Feedback with bar graphs and insights.

File Upload Support (Text file input for assessment).

Modern UI with a custom background and glassmorphism design.

Final Verdict on essay quality.

🛠️ Tech Stack

Frontend: HTML, CSS, JavaScript (Bootstrap for styling).

Backend: Flask (Python).

AI Model: PassbyGrocer/bert_bilstm_crf-ner-weibo from Hugging Face.

Libraries Used:

numpy, pandas

textstat (for readability analysis)

matplotlib (for visualization)

torch (for AI-based scoring)

📂 Project Structure

project-folder/
│── static/
│   ├── styles.css      # Custom CSS for UI styling
│   ├── background.jpg  # Custom background image
│── templates/
│   ├── index.html      # Home page
│   ├── results.html    # Display results & score visualization
│── app.py              # Flask application
│── model.py            # AI model integration
│── requirements.txt    # Required dependencies
│── README.md           # Project documentation

🔧 Installation & Setup

Clone the Repository

git clone https://github.com/your-repo/essay-assessment.git
cd essay-assessment

Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install Dependencies

pip install -r requirements.txt

Run the Flask App

python app.py

Open in Browser
Navigate to http://127.0.0.1:5000/ to access the essay submission page.

📊 Scoring System Breakdown

Attribute

Description

Grammar

Checks for grammatical errors and sentence correctness.

Clarity

Measures readability and complexity using Flesch Reading Ease.

Structure

Evaluates coherence, paragraph organization, and flow.

Vocabulary

Assesses word usage, richness, and appropriateness.

🖼️ UI Preview



🤖 AI Model Used

This project utilizes the BERT-BiLSTM-CRF model for Named Entity Recognition (NER) to assess text quality.

📌 Future Enhancements

Improve AI model accuracy.

Add plagiarism detection.

Enable multi-language essay evaluation.
