import joblib
import numpy as np
import shap
import pandas as pd
from catboost import CatBoostRegressor

try:
    salary_model = CatBoostRegressor()
    salary_model.load_model("models_&_files/catboost_salarypred.cbm")
    salary_vectorizer = joblib.load('models_&_files/tfidf_vectorizer.pkl')
except FileNotFoundError:
    print("Warning: Model artifacts not found. Ensure .pkl files are in the /models_&_files directory.")


def predict_salary(resume_text: str) -> float:
    if not resume_text or not resume_text.strip():
        return 0.0

    text_vector = salary_vectorizer.transform([resume_text])
    log_salary_pred = salary_model.predict(text_vector)[0]
    actual_salary = np.expm1(log_salary_pred)

    return float(actual_salary)


def get_salary_shap_data(skills_string, max_display=8):
    X_matrix = salary_vectorizer.transform([skills_string])
    feature_names = salary_vectorizer.get_feature_names_out()

    X_dense = X_matrix.toarray()
    feature_presence = np.ravel(X_dense[0])

    explainer = shap.TreeExplainer(salary_model)
    shap_values = explainer(X_dense)

    vals = shap_values.values[1] if isinstance(shap_values.values, list) else shap_values.values
    bases = shap_values.base_values[1] if isinstance(shap_values.base_values, list) else shap_values.base_values

    log_contributions = np.ravel(vals[0])
    base_log = float(np.ravel(bases)[0])
    final_log = base_log + log_contributions.sum()

    base_dollar = float(np.exp(base_log))
    final_dollar = float(np.exp(final_log))
    dollar_diff = final_dollar - base_dollar
    log_diff = final_log - base_log

    if abs(log_diff) < 1e-6:
        dollar_contributions = np.zeros_like(log_contributions)
    else:
        dollar_contributions = (log_contributions / log_diff) * dollar_diff

    df_shap = pd.DataFrame({
        'Skill': feature_names,
        'Dollar_Impact': dollar_contributions,
        'Is_Present': feature_presence > 0
    })

    df_shap = df_shap[df_shap['Dollar_Impact'] != 0]

    df_present = df_shap[df_shap['Is_Present']].copy()
    df_absent = df_shap[~df_shap['Is_Present']].copy()

    df_present['Skill'] = df_present['Skill'].str.title()
    df_present['Abs_Impact'] = df_present['Dollar_Impact'].abs()
    df_present = df_present.sort_values(by='Abs_Impact', ascending=False)

    top_skills = df_present.head(max_display).copy()

    other_present = df_present.iloc[max_display:]
    if not other_present.empty:
        other_sum = float(other_present['Dollar_Impact'].sum())
        other_count = len(other_present)
        if round(other_sum) != 0:
            other_row = pd.DataFrame({
                'Skill': [f"{other_count} Other Extracted Competencies"],
                'Dollar_Impact': [other_sum],
                'Abs_Impact': [abs(other_sum)]
            })
            top_skills = pd.concat([top_skills, other_row], ignore_index=True)  

    absent_sum = float(df_absent['Dollar_Impact'].sum())
    if round(absent_sum) != 0:
        absent_row = pd.DataFrame({
            'Skill': ["Penalty for Missing Competencies"],
            'Dollar_Impact': [absent_sum],
            'Abs_Impact': [abs(absent_sum)]
        })
        top_skills = pd.concat([top_skills, absent_row], ignore_index=True)

    top_skills = top_skills.sort_values(by='Abs_Impact', ascending=True)

    return base_dollar, final_dollar, top_skills
