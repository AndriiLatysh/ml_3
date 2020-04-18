import joblib
import math


qualification_model = joblib.load("models/qualification_by_two_grades_model.joblib")

while True:
    technical_grade = int(input("Technical grade: "))
    english_grade = int(input("English grade: "))

    grades = [[technical_grade, english_grade]]
    model_response = qualification_model.predict(grades)[0]
    model_confidence = qualification_model.predict_proba(grades)[0][1]

    text_response = ""
    if model_response == 0:
        text_response = "fails"
    elif model_response == 1:
        text_response = "passes"

    confidence_response = math.floor(abs(200 * model_confidence - 100))

    print("This candidate {} with the {}% confidence.".format(text_response, confidence_response))
