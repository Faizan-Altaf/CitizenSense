# CitizenSense — AI-Powered Public Feedback Triage

CitizenSense is a machine-learning-powered Streamlit application designed to help organize and prioritize public complaints. It analyzes complaint text, predicts its category and sentiment, estimates priority, and suggests the relevant department and next action.

## Features

- **Complaint classification:** Categorizes complaints into areas such as Roads / Infrastructure, Water Supply, Electricity, Sanitation, Health, and Education.
- **Sentiment analysis:** Predicts the tone of a complaint, such as Negative, Neutral, Positive, or Friendly.
- **Priority prediction:** Classifies complaints by urgency: Low, Medium, High, or Urgent.
- **Safety escalation:** Includes configurable rules to escalate complaints containing recognized danger-related terms.
- **Human review indicators:** Flags predictions when model confidence falls below a configurable threshold.
- **Complaint insights:** Displays category, sentiment, and priority summaries for the current session.
- **CSV batch analysis:** Upload a CSV containing complaint text, analyze multiple records, and download the results.
- **Suggested actions:** Recommends a department and an appropriate next step for each complaint.
- **Roman Urdu and English examples:** Includes sample complaints to demonstrate the application.

## Technology Stack

- Python
- Streamlit
- Scikit-learn
- TF-IDF text features
- Logistic Regression / trained classification pipelines
- Pandas and NumPy
- Joblib
- Altair

## Project Structure

```text
CitizenSense/
├── citizensense_app.py
├── requirements.txt
├── README.md
└── models/
    ├── tfidf_logistic_regression.joblib
    ├── tfidf_sentiment_classifier.joblib
    └── priority_classifier.joblib
```

The three trained model files must be available in the `models/` directory.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Faizan-Altaf/CitizenSense.git
cd CitizenSense
```

### 2. Create a virtual environment (recommended)

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run citizensense_app.py
```

Open the local URL shown in the terminal, usually:

`http://localhost:8501`

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Visit https://share.streamlit.io/
3. Connect your GitHub account and select `Faizan-Altaf/CitizenSense`.
4. Select the `main` branch.
5. Set the main file path to `citizensense_app.py`.
6. Deploy the application.

Ensure that the required model files are accessible to the deployed application and that `requirements.txt` contains compatible dependencies.

## Required Model Files

CitizenSense expects the following files:

- `tfidf_logistic_regression.joblib` — category classification model.
- `tfidf_sentiment_classifier.joblib` — sentiment classification model.
- `priority_classifier.joblib` — priority classification model.

These files are loaded at application startup. If they are missing, the application cannot perform predictions. Use the same compatible Scikit-learn environment used to train the models.

## Usage

1. Open CitizenSense.
2. Enter a complaint or select a sample.
3. Click **Analyze complaint**.
4. Review the predicted category, sentiment, priority, suggested department, and recommended action.
5. Use the CSV analysis tab to process multiple complaints.
6. Review session insights to understand the distribution of analyzed complaints.

## Responsible Use

CitizenSense is a decision-support prototype, not a replacement for human judgment. Model predictions and confidence scores may be incorrect. Human review is especially important for urgent, ambiguous, or safety-critical complaints. The safety escalation feature relies on configured text rules and should not be treated as a complete emergency detection system.

Avoid uploading private citizen information or sensitive complaint data to public repositories or demonstration environments.

## Author

**Faizan Altaf**

GitHub: https://github.com/Faizan-Altaf

## License

No license has been specified yet. Unless a license is added, the repository should not be assumed to grant permission for reuse or redistribution.
