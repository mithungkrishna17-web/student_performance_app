# 🎓 Student Performance Predictor

Predicts whether a student will **pass or fail** from study hours, attendance,
socioeconomic status, previous grades and other factors, so institutions can
identify **at-risk students** early and offer timely support.

Three models are trained and compared — **Decision Tree**, **Random Forest**
and **Support Vector Machine (SVM)** — and evaluated on **accuracy, precision,
recall and F1 score** (plus ROC AUC).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Trained models are included; if `models/` is
missing, the app generates data and trains automatically on first launch.

To retrain from the command line:

```bash
python generate_data.py     # (optional) regenerate the synthetic dataset
python train.py             # or: python train.py path/to/your_data.csv
```

## App features

| Tab | What it does |
|---|---|
| 🔮 Predict a student | Enter a student's roll number, name, class and details → Pass/Fail, probability of failing, risk gauge, all-model comparison, and recommended support actions; keeps a downloadable list of students checked this session |
| 📋 Batch screening | Upload a class CSV → every student ranked by risk, filter by class, chart of at-risk students per class, downloadable results |
| 📊 Model evaluation | Metrics table and chart for all 3 models, confusion matrix, feature importance, tuned hyperparameters, retrain on your own data |
| 🔍 Data explorer | Distribution of each feature split by Pass/Fail |

Choose the model in the sidebar (defaults to the best by F1).

## Dataset

`data/students.csv` — 2,000 synthetic students (70% pass / 30% fail):

| Column | Type | Values |
|---|---|---|
| roll_no | number | roll number within the class (display only) |
| name | text | student name (display only, not used by the models) |
| class | text | e.g. 10-A (display only, used for filtering) |
| study_hours | number | hours per week (0–40) |
| attendance | number | % (30–100) |
| previous_grade | number | 0–100 |
| socioeconomic_status | category | Low, Medium, High |
| parental_education | category | No Formal, High School, Bachelor, Master+ |
| internet_access | category | Yes, No |
| extracurricular | category | Yes, No |
| sleep_hours | number | hours per night |
| tutoring_sessions | integer | sessions per week |
| result | target | Pass, Fail |

To use real student records, provide a CSV with the same column names (via
`python train.py my_data.csv` or the **Retrain** section in the app).

## Methodology

- 80/20 stratified train/test split.
- Preprocessing pipeline: numeric features standardised, ordinal encoding for
  SES and parental education, one-hot for yes/no features.
- `class_weight="balanced"` on every model to counter class imbalance.
- Hyperparameters tuned with `GridSearchCV`, 5-fold stratified CV, scoring F1.
- **"Fail" is the positive class** — recall measures how many at-risk students
  the model catches, which matters most for intervention.

## Results (test set, 400 students)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Decision Tree | 0.800 | 0.643 | 0.750 | 0.692 | 0.835 |
| Random Forest | **0.858** | **0.760** | 0.767 | 0.763 | 0.908 |
| **SVM** | 0.848 | 0.712 | **0.825** | **0.764** | **0.926** |

SVM has the best F1 and recall, catching 82.5% of failing students. Random
Forest has the highest accuracy and precision (fewest false alarms). Previous grade, study hours
and attendance are the strongest predictors.

## Project structure

```
student_performance_app/
├── app.py              # Streamlit web application
├── train.py            # trains & evaluates DT, RF, SVM; saves models + metrics
├── generate_data.py    # creates the synthetic dataset
├── requirements.txt
├── data/students.csv
└── models/             # *.joblib pipelines + metrics.json
```
