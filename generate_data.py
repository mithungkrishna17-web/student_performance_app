"""
Generate a realistic synthetic student dataset.

Replace data/students.csv with your institution's real data (same column names)
to train on actual records.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 2000


def generate(n: int = N) -> pd.DataFrame:
    ses = RNG.choice(["Low", "Medium", "High"], size=n, p=[0.3, 0.5, 0.2])
    ses_score = pd.Series(ses).map({"Low": 0, "Medium": 1, "High": 2}).to_numpy()

    study_hours = np.clip(RNG.normal(12 + 2 * ses_score, 5, n), 0, 40).round(1)   # per week
    attendance = np.clip(RNG.normal(78 + 4 * ses_score, 12, n), 30, 100).round(1)  # %
    previous_grade = np.clip(RNG.normal(62 + 4 * ses_score, 14, n), 20, 100).round(1)
    parental_education = RNG.choice(
        ["No Formal", "High School", "Bachelor", "Master+"], size=n, p=[0.1, 0.4, 0.35, 0.15]
    )
    internet_access = RNG.choice(["Yes", "No"], size=n, p=[0.8, 0.2])
    extracurricular = RNG.choice(["Yes", "No"], size=n, p=[0.45, 0.55])
    sleep_hours = np.clip(RNG.normal(7, 1.2, n), 3, 11).round(1)
    tutoring_sessions = RNG.poisson(1.5, n)

    pe_score = pd.Series(parental_education).map(
        {"No Formal": 0, "High School": 1, "Bachelor": 2, "Master+": 3}
    ).to_numpy()

    # latent "final score" driving pass/fail
    score = (
        0.45 * previous_grade
        + 0.9 * study_hours
        + 0.30 * attendance
        + 2.5 * ses_score
        + 1.5 * pe_score
        + 3.0 * (internet_access == "Yes")
        + 1.0 * (extracurricular == "Yes")
        - 2.0 * np.abs(sleep_hours - 7.5)
        + 1.5 * tutoring_sessions
        + RNG.normal(0, 6, n)
    )
    result = np.where(score >= np.percentile(score, 30), "Pass", "Fail")

    first = ["Aarav", "Vivaan", "Aditya", "Arjun", "Rohan", "Karthik", "Rahul", "Siddharth",
             "Pranav", "Nikhil", "Ananya", "Diya", "Priya", "Sneha", "Kavya", "Meera",
             "Aarthi", "Lakshmi", "Divya", "Pooja", "Ishaan", "Varun", "Neha", "Shreya"]
    last = ["Sharma", "Reddy", "Nair", "Iyer", "Kumar", "Patel", "Rao", "Menon", "Singh",
            "Gupta", "Pillai", "Das", "Joshi", "Shetty", "Krishnan", "Verma"]
    names = [f"{RNG.choice(first)} {RNG.choice(last)}" for _ in range(n)]
    classes = [f"{g}-{s}" for g, s in zip(RNG.choice([9, 10, 11, 12], n), RNG.choice(list("ABC"), n))]

    # roll numbers run 1, 2, 3 … within each class
    roll_no = pd.Series(classes).groupby(classes).cumcount().to_numpy() + 1

    return pd.DataFrame({
        "roll_no": roll_no,
        "name": names,
        "class": classes,
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_grade": previous_grade,
        "socioeconomic_status": ses,
        "parental_education": parental_education,
        "internet_access": internet_access,
        "extracurricular": extracurricular,
        "sleep_hours": sleep_hours,
        "tutoring_sessions": tutoring_sessions,
        "result": result,
    })


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "students.csv"
    out.parent.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(df["result"].value_counts())
