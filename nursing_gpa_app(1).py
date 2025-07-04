# This will be your Streamlit app to analyze nursing program eligibility

import streamlit as st
from datetime import datetime
from typing import List, Dict
import pandas as pd
import json
import pdfplumber
import re
import subprocess
import tempfile

# ==== GPA Utilities ====
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

def ollama_fallback_parse(text: str) -> List[Dict]:
    prompt = f"""Extract course data from this transcript text. Return a JSON array. Each item should have:
    - course_code
    - title
    - credits
    - grade
    - date (YYYY-MM)

{text}
"""
    try:
        command = ["ollama", "run", "mistral"]
        with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.txt') as tmp:
            tmp.write(prompt)
            tmp.flush()
            result = subprocess.run(command, input=prompt.encode('utf-8'), capture_output=True, timeout=60)
            output = result.stdout.decode('utf-8')
            parsed = json.loads(output)
            return parsed
    except Exception as e:
        st.error(f"Ollama fallback failed: {e}")
        return []

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
        current_term = None
        term_pattern = re.compile(r"^(Fall|Spring|Summer|Winter)\s+\d{4}$")
        course_patterns = [
            re.compile(r"^([A-Z]{3,4})\s+(\d{3})\s+(.+?)\s{2,}(\d\.\d{2})\s+([A-F][+-]?)$"),
            re.compile(r"^([A-Z]{2,4}\s?\d{3})\s+(.+?)\s+(\d\.\d{2})\s+([A-F][+-]?)$")
        ]

        for line in lines:
            line = line.strip()
            if term_pattern.match(line):
                current_term = line
                continue
            for pattern in course_patterns:
                match = pattern.match(line)
                if match:
                    try:
                        if len(match.groups()) == 5:
                            subject = match.group(1)
                            catalog = match.group(2)
                            course_code = f"{subject} {catalog}"
                            title = match.group(3).strip()
                            credits = float(match.group(4))
                            grade = match.group(5)
                        elif len(match.groups()) == 4:
                            course_code = match.group(1)
                            title = match.group(2).strip()
                            credits = float(match.group(3))
                            grade = match.group(4)
                        else:
                            continue
                        date = datetime.strptime(current_term, "%B %Y") if current_term else datetime(2000, 1, 1)
                        data.append({
                            'course_code': course_code,
                            'title': title,
                            'credits': credits,
                            'grade': grade,
                            'date': date,
                            'matches': None
                        })
                        break
                    except:
                        continue

        if not data:
            st.warning("No structured data found with regex. Attempting Ollama fallback...")
            parsed = ollama_fallback_parse(text)
            for item in parsed:
                try:
                    data.append({
                        'course_code': item['course_code'],
                        'title': item['title'],
                        'credits': float(item['credits']),
                        'grade': item['grade'],
                        'date': pd.to_datetime(item['date']),
                        'matches': None
                    })
                except:
                    continue

        df = pd.DataFrame(data)

    st.subheader("Parsed Transcript Data")
    st.dataframe(df)

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
