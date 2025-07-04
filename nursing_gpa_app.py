# This will be your Streamlit app to analyze nursing program eligibility

import streamlit as st
from datetime import datetime
from typing import List, Dict
import pandas as pd
import json
import pdfplumber
import re

# ==== GPA Utilities (pulled from module) ====
GRADE_POINTS = {
    'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0,
    'B-': 2.7, 'C+': 2.3, 'C': 2.0, 'C-': 1.7,
    'D+': 1.3, 'D': 1.0, 'F': 0.0
}

def calculate_gpa(courses: List[Dict]) -> float:
    total_points = 0.0
    total_credits = 0
    for course in courses:
        grade = GRADE_POINTS.get(course['grade'])
        if grade is not None:
            total_points += grade * course['credits']
            total_credits += course['credits']
    return round(total_points / total_credits, 2) if total_credits else None

def calculate_last_60_gpa(courses: List[Dict]) -> float:
    courses = sorted(courses, key=lambda x: x['date'], reverse=True)
    total_credits = 0
    total_points = 0.0
    for course in courses:
        if total_credits + course['credits'] <= 60:
            total_credits += course['credits']
            total_points += GRADE_POINTS[course['grade']] * course['credits']
        else:
            needed = 60 - total_credits
            fraction = needed / course['credits']
            total_points += GRADE_POINTS[course['grade']] * course['credits'] * fraction
            total_credits = 60
            break
    return round(total_points / 60, 2) if total_credits == 60 else None

def calculate_prereq_gpa(courses: List[Dict], matched_prereqs: List[str]) -> float:
    total_points = 0.0
    total_credits = 0
    for course in courses:
        if course.get('matches') in matched_prereqs:
            grade = GRADE_POINTS.get(course['grade'])
            if grade is not None:
                total_points += grade * course['credits']
                total_credits += course['credits']
    return round(total_points / total_credits, 2) if total_credits else None

# ==== Streamlit UI ====
st.set_page_config(page_title="Nursing GPA Analyzer", layout="wide")
st.title("🩺 Nursing Program Eligibility Checker")

st.write("Upload your transcript data (PDF, CSV, or JSON format)")

uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'csv', 'json'])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        data = json.load(uploaded_file)
        df = pd.DataFrame(data)
    elif uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        st.text_area("Extracted Transcript Text (for dev/debug)", text, height=300)
        
        lines = text.split("\n")
        data = []
        course_pattern = re.compile(r"([A-Z]{2,}\d{3})\s+(.+?)\s+(\d+\.\d)\s+([A-F][+-]?)\s+(\w+\s\d{4}|\d{4}-\d{2}-\d{2})")

        for line in lines:
            match = course_pattern.search(line)
            if match:
                course_code = match.group(1)
                title = match.group(2)
                credits = float(match.group(3))
                grade = match.group(4)
                raw_date = match.group(5)
                try:
                    date = datetime.strptime(raw_date, "%Y-%m-%d")
                except:
                    try:
                        date = datetime.strptime(raw_date, "%B %Y")
                    except:
                        date = datetime(2000, 1, 1)  # Fallback placeholder
                data.append({
                    'course_code': course_code,
                    'title': title,
                    'credits': credits,
                    'grade': grade,
                    'date': date,
                    'matches': None
                })
        df = pd.DataFrame(data)

    st.subheader("Parsed Transcript Data")
    st.dataframe(df)

    # Convert to required format
    courses = []
    for _, row in df.iterrows():
        try:
            courses.append({
                'course_code': row['course_code'],
                'grade': row['grade'],
                'credits': float(row['credits']),
                'date': pd.to_datetime(row['date']),
                'matches': row.get('matches', None)
            })
        except:
            continue

    # Prerequisite match list (placeholder)
    prereqs = [
        'Anatomy & Physiology I', 'Anatomy & Physiology II', 'Microbiology',
        'Chemistry', 'Nutrition', 'Statistics', 'Human Growth & Development'
    ]

    st.subheader("📊 GPA Results")

    cum_gpa = calculate_gpa(courses)
    last_60 = calculate_last_60_gpa(courses)
    prereq_gpa = calculate_prereq_gpa(courses, prereqs)

    st.markdown(f"**Cumulative GPA**: `{cum_gpa}`")
    st.markdown(f"**Last 60 Credits GPA**: `{last_60}`")
    st.markdown(f"**Prerequisite GPA**: `{prereq_gpa}`")

    st.subheader("📝 Recommendation")
    if cum_gpa and cum_gpa >= 3.0 and last_60 >= 3.0 and prereq_gpa >= 3.0:
        st.success("You're likely eligible for the University of Arizona's Nursing MS Program. Hell yeah!")
    else:
        st.error("You may need to improve one or more GPA metrics or complete missing prerequisites.")
