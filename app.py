import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import io
import numpy as np

# --- APP CONFIGURATION & ONE-TIME SETUP ---

st.set_page_config(
    layout="wide",
    page_title="U of A Nursing Transcript Analyzer",
    page_icon="⚕️"
)

# Use caching for expensive operations like loading the AI model
@st.cache_resource
def load_ai_model():
    """Loads the AI model once and caches it."""
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_ai_model()

# --- CONSTANTS & DATABASES ---

# Master list of U of A MEPN prerequisites. In a real app, this could be a separate config file.
UA_PREREQS = {
    "Human Anatomy & Physiology I w/ Lab": {
        "ua_course": "PSIO 201",
        "credits": 4,
        "description": "Integrated study of the structure and function of the human body. Includes cells, tissues, and the integumentary, skeletal, muscular, and nervous systems. Laboratory component required."
    },
    "Human Anatomy & Physiology II w/ Lab": {
        "ua_course": "PSIO 202",
        "credits": 4,
        "description": "Continuation of PSIO 201. Covers the endocrine, cardiovascular, lymphatic, respiratory, digestive, urinary, and reproductive systems. Laboratory component required."
    },
    "Microbiology w/ Lab": {
        "ua_course": "MIC 205A/L",
        "credits": 4,
        "description": "Fundamentals of microbiology, including microbial cell structure and function, metabolism, genetics, and the role of microorganisms in disease and the environment. Laboratory component required."
    },
    "Chemistry w/ Lab": {
        "ua_course": "CHEM 151 or higher",
        "credits": 4,
        "description": "Introduction to the basic principles of chemistry, including atomic structure, chemical bonding, stoichiometry, states of matter, and solutions. Laboratory component required."
    },
    "Statistics": {
        "ua_course": "MATH 163 or higher",
        "credits": 3,
        "description": "Introduction to statistical reasoning, including data collection, descriptive statistics, probability, and inference. Topics include hypothesis testing and confidence intervals."
    },
    "Lifespan Development": {
        "ua_course": "FSHD 117",
        "credits": 3,
        "description": "Survey of human development from conception through death, covering physical, cognitive, social, and emotional domains across the lifespan."
    },
}

GRADE_TO_POINTS = {
    'A+': 4.0, 'A': 4.0, 'A-': 3.7,
    'B+': 3.3, 'B': 3.0, 'B-': 2.7,
    'C+': 2.3, 'C': 2.0, 'C-': 1.7,
    'D+': 1.3, 'D': 1.0, 'F': 0.0,
}

# --- HELPER FUNCTIONS ---

def extract_text_from_pdf(pdf_file):
    """Extracts raw text from an uploaded PDF file."""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def parse_courses_from_text(text):
    """
    Parses extracted text to find courses, credits, and grades.
    This is a simplified regex and will need adjustment for different transcript formats.
    """
    # Regex to find a course code (e.g., CHEM 101), a title, credits, and a grade.
    # This is HIGHLY dependent on transcript format.
    pattern = re.compile(
        r'([A-Z]{2,4}\s*\d{3,4}[A-Z]?)\s+(.*?)\s+(\d\.\d+)\s+([A-Z][+-]?)'
    )
    matches = pattern.findall(text)
    
    if not matches:
        st.warning("Could not automatically parse courses. Please enter them manually below.")
        return pd.DataFrame(columns=['Course_Code', 'Title', 'Credits', 'Grade'])

    df = pd.DataFrame(matches, columns=['Course_Code', 'Title', 'Credits', 'Grade'])
    
    # Data cleaning
    df['Credits'] = pd.to_numeric(df['Credits'])
    df['Grade_Points'] = df['Grade'].map(GRADE_TO_POINTS)
    
    # Add placeholder semester data for last 60 credits calculation demo
    # In a real app, this would also need to be parsed.
    num_courses = len(df)
    semesters = ['Fall 2023'] * (num_courses // 4) + \
                ['Spring 2023'] * (num_courses // 4) + \
                ['Fall 2022'] * (num_courses // 4) + \
                ['Spring 2022'] * (num_courses - 3 * (num_courses // 4))
    df['Semester'] = sorted(semesters, reverse=True)

    return df

@st.cache_data(show_spinner="Searching for course description...")
def get_course_description_from_web(university, course_code):
    """
    A best-effort attempt to scrape a course description.
    This is fragile and a manual fallback is essential.
    """
    try:
        query = f'"{university}" "{course_code}" course description'
        search_url = f"https://www.google.com/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # This is a heuristic: find the first search result link and try to find the description there
        # A more robust solution might use a dedicated API or more advanced scraping.
        first_link = soup.find('a', href=re.compile(r'/url\?q='))
        if not first_link:
            return "Could not find a link to the course catalog."

        page_url = first_link['href'].split('/url?q=')[1].split('&sa=')[0]
        page_response = requests.get(page_url, headers=headers, timeout=5)
        page_soup = BeautifulSoup(page_response.text, 'html.parser')
        
        page_text = page_soup.get_text()
        # Find the course code and get the text following it.
        match = re.search(re.escape(course_code) + r'[\s\S]{1,500}', page_text, re.IGNORECASE)
        if match:
            # Clean up the found text
            description = ' '.join(match.group(0).split())
            return description[:400] + "..." # Truncate for display
        return "Found the course page, but could not automatically extract a description."
        
    except Exception as e:
        return f"Web scraping failed: {e}. Please find and paste the description manually."

def calculate_similarity(desc1, desc2):
    """Calculates the semantic similarity between two course descriptions using the AI model."""
    if not desc1 or not desc2:
        return 0.0
    embedding1 = model.encode(desc1, convert_to_tensor=True)
    embedding2 = model.encode(desc2, convert_to_tensor=True)
    cosine_score = util.cos_sim(embedding1, embedding2)
    return cosine_score.item()

def calculate_last_60_gpa(df):
    """Calculates the GPA for the student's most recent 60 credit hours."""
    if df.empty or 'Semester' not in df.columns:
        return 0.0, pd.DataFrame()
        
    # Sort by a semester proxy (this is a simplification)
    df_sorted = df.sort_values(by='Semester', ascending=False, kind='mergesort')
    
    credits_sum = 0
    courses_in_gpa = []
    
    for _, row in df_sorted.iterrows():
        if credits_sum + row['Credits'] <= 65: # Allow going slightly over to include the full course
            credits_sum += row['Credits']
            courses_in_gpa.append(row)
        else:
            break
            
    if not courses_in_gpa:
        return 0.0, pd.DataFrame()

    last_60_df = pd.DataFrame(courses_in_gpa)
    
    if last_60_df.empty or 'Credits' not in last_60_df.columns or 'Grade_Points' not in last_60_df.columns:
        return 0.0, pd.DataFrame()
        
    total_quality_points = (last_60_df['Credits'] * last_60_df['Grade_Points']).sum()
    total_credits = last_60_df['Credits'].sum()
    
    gpa = total_quality_points / total_credits if total_credits > 0 else 0.0
    return gpa, last_60_df


# --- MAIN APP INTERFACE ---

st.title("⚕️ U of A MEPN Transcript Analyzer")
st.markdown("An internal tool for admissions advisors to streamline prerequisite and GPA evaluation.")
st.warning("⚠️ **Confidentiality Notice:** This tool processes FERPA-protected data. Do not use on public networks or share the results outside of the admissions committee.")

# Initialize session state to hold matches
if 'prereq_matches' not in st.session_state:
    st.session_state.prereq_matches = {}

# --- SIDEBAR FOR UPLOAD AND CONTROLS ---

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("1. Upload Transcript PDF", type="pdf")
    university_name = st.text_input("2. Enter Applicant's University Name", placeholder="e.g., Arizona State University")

    if st.button("Reset Analysis"):
        st.session_state.prereq_matches = {}
        st.rerun()

# --- MAIN CONTENT AREA ---

if uploaded_file and university_name:
    raw_text = extract_text_from_pdf(uploaded_file)
    
    if raw_text:
        st.subheader("Extracted & Parsed Transcript Data")
        st.info("The table below shows the courses parsed from the PDF. **Please review and correct any errors** before proceeding. The accuracy of the analysis depends on this data.")
        
        parsed_df = parse_courses_from_text(raw_text)
        
        # The editable dataframe is the key to handling parsing errors
        edited_df = st.data_editor(
            parsed_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Grade_Points": st.column_config.NumberColumn("Grade Points", format="%.1f")
            }
        )
        
        st.divider()

        tab1, tab2 = st.tabs(["➡️ Prerequisite Analysis Workflow", "📊 Final Summary Report"])

        # --- PREREQUISITE ANALYSIS TAB ---
        with tab1:
            st.header("Prerequisite Matching")
            st.markdown("For each U of A requirement, select the applicant's course that may fulfill it.")

            for req_name, req_details in UA_PREREQS.items():
                with st.expander(f"**{req_name}** ({req_details['ua_course']})", expanded=False):
                    
                    # Create a unique key for each set of widgets inside the expander
                    select_key = f"select_{req_details['ua_course']}"
                    
                    # Pre-populate selectbox if a match was already confirmed
                    current_match_code = st.session_state.prereq_matches.get(req_name, {}).get('Course_Code')
                    
                    options = ['- Not Selected -'] + edited_df['Course_Code'].tolist()
                    try:
                        current_index = options.index(current_match_code) if current_match_code else 0
                    except ValueError:
                        current_index = 0

                    selected_course_code = st.selectbox(
                        "Select Applicant's Course",
                        options,
                        index=current_index,
                        key=select_key,
                        help="Choose the course from the parsed list that you believe matches this prerequisite."
                    )

                    if selected_course_code != '- Not Selected -':
                        student_course = edited_df[edited_df['Course_Code'] == selected_course_code].iloc[0]
                        st.write(f"**Selected:** {student_course['Title']} | **Credits:** {student_course['Credits']} | **Grade:** {student_course['Grade']}")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**U of A Description ({req_details['ua_course']})**")
                            st.write(req_details['description'])
                        
                        with col2:
                            st.warning(f"**Applicant Course Description ({university_name})**")
                            
                            # Attempt to scrape, with manual override
                            scraped_desc = get_course_description_from_web(university_name, student_course['Course_Code'])
                            manual_desc = st.text_area("Find and paste description here if needed:", value=scraped_desc, height=150, key=f"desc_{req_details['ua_course']}")
                        
                        # Perform comparison
                        similarity = calculate_similarity(req_details['description'], manual_desc)
                        st.progress(similarity, text=f"Semantic Similarity Score: {similarity:.0%}")
                        if similarity > 0.7:
                            st.success("High Similarity: Strong potential match.")
                        elif similarity > 0.5:
                            st.warning("Moderate Similarity: Advisor review is critical.")
                        else:
                            st.error("Low Similarity: Unlikely to be a match.")

                        if st.button("Confirm this Match", key=f"confirm_{req_details['ua_course']}"):
                            st.session_state.prereq_matches[req_name] = student_course.to_dict()
                            st.toast(f"Match for {req_name} confirmed!", icon="✅")
                            st.rerun()

        # --- FINAL SUMMARY REPORT TAB ---
        with tab2:
            st.header("Evaluation Summary")

            if not st.session_state.prereq_matches:
                st.warning("No prerequisite matches have been confirmed yet. Go to the 'Analysis Workflow' tab to begin.")
            else:
                st.subheader("Confirmed Prerequisite Matches")
                
                # Prepare data for display
                summary_data = []
                matched_courses_df = pd.DataFrame(st.session_state.prereq_matches.values())

                for req_name, req_details in UA_PREREQS.items():
                    match = st.session_state.prereq_matches.get(req_name)
                    if match:
                        summary_data.append({
                            "U of A Requirement": req_name,
                            "Status": "✅ Met",
                            "Applicant Course": f"{match['Course_Code']} - {match['Title']}",
                            "Grade": match['Grade'],
                            "Credits": match['Credits']
                        })
                    else:
                        summary_data.append({
                            "U of A Requirement": req_name,
                            "Status": "❌ Missing",
                            "Applicant Course": "---",
                            "Grade": "---",
                            "Credits": "---"
                        })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # GPA Calculations
                st.subheader("GPA Calculations")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Prerequisite GPA**")
                    if not matched_courses_df.empty and 'Grade_Points' in matched_courses_df.columns:
                        prereq_gpa = (matched_courses_df['Credits'] * matched_courses_df['Grade_Points']).sum() / matched_courses_df['Credits'].sum()
                        st.metric(label="Calculated Prereq GPA", value=f"{prereq_gpa:.3f}")
                        if prereq_gpa < 3.0:
                             st.error("GPA is below the typical 3.0 minimum.")
                        else:
                             st.success("GPA meets the typical 3.0 minimum.")
                    else:
                        st.metric(label="Calculated Prereq GPA", value="N/A")

                with col2:
                    st.markdown("**Last 60 Credits GPA**")
                    last_60_gpa, last_60_df = calculate_last_60_gpa(edited_df)
                    st.metric(label="Calculated Last 60 Credits GPA", value=f"{last_60_gpa:.3f}")
                    with st.expander("Show courses included in Last 60 Credits GPA"):
                        st.dataframe(last_60_df, use_container_width=True, hide_index=True)

else:
    st.info("Please upload a transcript PDF and enter the university name in the sidebar to begin analysis.")