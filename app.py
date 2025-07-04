import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import pytesseract
import io

# --- APP CONFIGURATION & ONE-TIME SETUP ---
st.set_page_config(layout="wide", page_title="U of A Nursing Transcript Analyzer", page_icon="⚕️")

@st.cache_resource
def load_ai_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

model = load_ai_model()

# --- CONSTANTS & DATABASES ---
UA_PREREQS = {
    "Human Anatomy & Physiology I w/ Lab": {"ua_course": "PSIO 201", "credits": 4, "description": "Integrated study of the structure and function of the human body. Includes cells, tissues, and the integumentary, skeletal, muscular, and nervous systems. Laboratory component required."},
    "Human Anatomy & Physiology II w/ Lab": {"ua_course": "PSIO 202", "credits": 4, "description": "Continuation of PSIO 201. Covers the endocrine, cardiovascular, lymphatic, respiratory, digestive, urinary, and reproductive systems. Laboratory component required."},
    "Microbiology w/ Lab": {"ua_course": "MIC 205A/L", "credits": 4, "description": "Fundamentals of microbiology, including microbial cell structure and function, metabolism, genetics, and the role of microorganisms in disease and the environment. Laboratory component required."},
    "Chemistry w/ Lab": {"ua_course": "CHEM 151 or higher", "credits": 4, "description": "Introduction to the basic principles of chemistry, including atomic structure, chemical bonding, stoichiometry, states of matter, and solutions. Laboratory component required."},
    "Statistics": {"ua_course": "MATH 163 or higher", "credits": 3, "description": "Introduction to statistical reasoning, including data collection, descriptive statistics, probability, and inference. Topics include hypothesis testing and confidence intervals."},
    "Lifespan Development": {"ua_course": "FSHD 117", "credits": 3, "description": "Survey of human development from conception through death, covering physical, cognitive, social, and emotional domains across the lifespan."},
}
GRADE_TO_POINTS = {'A+': 4.0, 'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7, 'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0, 'F': 0.0}

# --- HELPER FUNCTIONS ---

def extract_text_from_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        try:
            with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc:
                text = "".join(page.get_text() for page in doc)
            return text
        except Exception as e:
            st.error(f"Error processing PDF {uploaded_file.name}: {e}")
            return ""
    elif file_extension in ['png', 'jpg', 'jpeg']:
        try:
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            st.error(f"Error processing image {uploaded_file.name}: {e}")
            return ""
    else:
        st.warning(f"Unsupported file type: {uploaded_file.name}. Please upload PDF, PNG, or JPG files.")
        return ""

def parse_courses_from_text(text):
    pattern = re.compile(r'([A-Z]{2,4}\s*\d{3,4}[A-Z]?)\s+(.*?)\s+(\d\.\d+)\s+([A-Z][+-]?)')
    matches = pattern.findall(text)
    if not matches: return pd.DataFrame()
    df = pd.DataFrame(matches, columns=['Course_Code', 'Title', 'Credits', 'Grade'])
    df['Credits'] = pd.to_numeric(df['Credits'])
    df['Grade_Points'] = df['Grade'].map(GRADE_TO_POINTS)
    return df

@st.cache_data(show_spinner="Searching for course description...")
def get_course_description_from_web(course_row):
    """
    A more reliable web scraper that uses DuckDuckGo and takes a full course row.
    """
    university = course_row['University']
    course_code = course_row['Course_Code']

    try:
        query = f'"{university}" "{course_code}" course catalog description'
        search_url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'}
        
        response = requests.post(search_url, data=params, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        search_results = soup.find_all('div', class_='result')
        if not search_results: return "Could not find any search results on DuckDuckGo."

        page_url = None
        for result in search_results:
            link_tag = result.find('a', class_='result__a')
            if link_tag and link_tag.has_attr('href'):
                page_url = link_tag['href']
                break
        
        if not page_url: return "Found search results, but could not extract a link."

        page_response = requests.get(page_url, headers=headers, timeout=10)
        page_response.raise_for_status()
        page_soup = BeautifulSoup(page_response.text, 'html.parser')
        page_text = page_soup.get_text()
        
        match_pos = page_text.lower().find(course_code.lower())
        if match_pos == -1: match_pos = page_text.lower().replace(" ", "").find(course_code.lower().replace(" ",""))
        
        if match_pos != -1:
            start_index = max(0, match_pos - 100)
            end_index = start_index + 600
            description = ' '.join(page_text[start_index:end_index].split())
            return description + "..."
        return "Found the course page, but could not automatically extract a description."
    except requests.exceptions.HTTPError as e: return f"Web search failed: HTTP Error {e.response.status_code}."
    except Exception as e: return f"An unexpected error occurred during web scraping: {e}."

def calculate_similarity(desc1, desc2):
    if not desc1 or not desc2: return 0.0
    embedding1 = model.encode(desc1, convert_to_tensor=True)
    embedding2 = model.encode(desc2, convert_to_tensor=True)
    return util.cos_sim(embedding1, embedding2).item()

# --- MAIN APP INTERFACE ---
st.title("⚕️ U of A MEPN Transcript Analyzer")
st.markdown("An internal tool for admissions advisors to streamline prerequisite and GPA evaluation.")

if 'prereq_matches' not in st.session_state:
    st.session_state.prereq_matches = {}

with st.sidebar:
    st.header("Controls")
    uploaded_files = st.file_uploader("1. Upload Transcript(s) (PDF, PNG, JPG)", type=['pdf', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)
    # --- CHANGED: Clarified role of the university name input ---
    university_name = st.text_input("2. Enter Primary/Default University Name", placeholder="e.g., Pima Community College")
    if st.button("Reset Analysis"):
        st.session_state.prereq_matches = {}
        st.rerun()

if uploaded_files and university_name:
    all_courses_list = []
    with st.spinner("Processing uploaded transcripts..."):
        for uploaded_file in uploaded_files:
            raw_text = extract_text_from_file(uploaded_file)
            if raw_text:
                parsed_df = parse_courses_from_text(raw_text)
                if not parsed_df.empty:
                    parsed_df['Source'] = uploaded_file.name
                    all_courses_list.append(parsed_df)

    if not all_courses_list:
        st.error("Could not parse any courses from the uploaded file(s). Please use the manual entry feature below.")
        # --- NEW: Added University column for manual entry ---
        edited_df = st.data_editor(pd.DataFrame(columns=['University', 'Source', 'Course_Code', 'Title', 'Credits', 'Grade', 'Grade_Points']), num_rows="dynamic", use_container_width=True)
    else:
        master_df = pd.concat(all_courses_list, ignore_index=True)
        # --- NEW: Add the default university name to all parsed courses ---
        master_df['University'] = university_name
        
        st.subheader("Parsed Transcript Data")
        st.info("The table below shows all courses parsed. **You can edit any cell, including the University, if a course is from a different institution.**")
        
        edited_df = st.data_editor(
            master_df, num_rows="dynamic", use_container_width=True,
            column_config={
                "University": st.column_config.TextColumn("University", help="The university where the course was taken."),
                "Grade_Points": st.column_config.NumberColumn("Grade Points", format="%.1f"),
                "Source": st.column_config.TextColumn("Source File")
            },
            # --- NEW: Added University to the column order ---
            column_order=['University', 'Source', 'Course_Code', 'Title', 'Credits', 'Grade', 'Grade_Points'],
            key="course_editor"
        )
    
    st.divider()
    tab1, tab2 = st.tabs(["➡️ Prerequisite Analysis Workflow", "📊 Final Summary Report"])
    
    with tab1:
        st.header("Prerequisite Matching")
        st.markdown("For each U of A requirement, select the applicant's course that may fulfill it.")
        for req_name, req_details in UA_PREREQS.items():
            with st.expander(f"**{req_name}** ({req_details['ua_course']})", expanded=False):
                select_key = f"select_{req_details['ua_course']}"
                current_match_code = st.session_state.prereq_matches.get(req_name, {}).get('Course_Code')
                options = ['- Not Selected -'] + edited_df['Course_Code'].tolist()
                try:
                    current_index = options.index(current_match_code) if current_match_code else 0
                except ValueError: current_index = 0

                selected_course_code = st.selectbox("Select Applicant's Course", options, index=current_index, key=select_key)
                if selected_course_code != '- Not Selected -':
                    # This now contains the correct university in the row
                    student_course = edited_df[edited_df['Course_Code'] == selected_course_code].iloc[0]
                    
                    st.write(f"**Selected:** {student_course['University']} - {student_course['Course_Code']} | **Credits:** {student_course['Credits']} | **Grade:** {student_course['Grade']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**U of A Description ({req_details['ua_course']})**")
                        st.write(req_details['description'])
                    with col2:
                        st.warning(f"**Applicant Course Description ({student_course['University']})**")
                        st.markdown("_Web search is a best-effort convenience. If it fails, please copy and paste the description manually._")
                        # --- CHANGED: Pass the entire row to the function ---
                        scraped_desc = get_course_description_from_web(student_course)
                        manual_desc = st.text_area("Find and paste description here if needed:", value=scraped_desc, height=150, key=f"desc_{req_details['ua_course']}")
                    
                    similarity_score = calculate_similarity(req_details['description'], manual_desc)
                    progress_value = max(0.0, min(1.0, similarity_score))
                    st.progress(progress_value, text=f"Semantic Similarity Score: {similarity_score:.0%}")
                    
                    if st.button("Confirm this Match", key=f"confirm_{req_details['ua_course']}"):
                        st.session_state.prereq_matches[req_name] = student_course.to_dict()
                        st.toast(f"Match for {req_name} confirmed!", icon="✅")
                        st.rerun()

    with tab2:
        st.header("Evaluation Summary")
        st.subheader("Confirmed Prerequisite Matches")
        if not st.session_state.prereq_matches:
            st.warning("No prerequisite matches have been confirmed yet.")
        else:
            summary_data = []
            for req_name, req_details in UA_PREREQS.items():
                match = st.session_state.prereq_matches.get(req_name)
                summary_data.append({
                    "U of A Requirement": req_name, "Status": "✅ Met" if match else "❌ Missing",
                    "Applicant Course": f"{match.get('University', '')} - {match.get('Course_Code', '')}" if match else "---",
                    "Grade": match.get('Grade', '---'), "Credits": match.get('Credits', '---')
                })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("GPA Calculations")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Prerequisite GPA**")
            if st.session_state.prereq_matches:
                matched_courses_df = pd.DataFrame(st.session_state.prereq_matches.values())
                if not matched_courses_df.empty and 'Grade_Points' in matched_courses_df.columns:
                    total_credits = pd.to_numeric(matched_courses_df['Credits']).sum()
                    total_quality_points = (pd.to_numeric(matched_courses_df['Credits']) * pd.to_numeric(matched_courses_df['Grade_Points'])).sum()
                    prereq_gpa = total_quality_points / total_credits if total_credits > 0 else 0.0
                    st.metric(label="Calculated Prereq GPA", value=f"{prereq_gpa:.3f}")
                    if prereq_gpa < 3.0: st.error("GPA is below the typical 3.0 minimum.")
                    else: st.success("GPA meets the typical 3.0 minimum.")
                else: st.metric(label="Calculated Prereq GPA", value="N/A")
            else: st.metric(label="Calculated Prereq GPA", value="N/A")

        with col2:
            st.markdown("**Recent Credits GPA (e.g., Last 60)**")
            st.info("Select the courses to include in the GPA calculation.")
            
            selection_df = edited_df.copy()
            selection_df['Select'] = False
            # Added University to the display here for clarity
            selection_df = selection_df[['Select', 'University', 'Source', 'Course_Code', 'Credits', 'Grade']]
            
            selected_courses_df = st.data_editor(
                selection_df, use_container_width=True, hide_index=True,
                disabled=['University', 'Source', 'Course_Code', 'Credits', 'Grade']
            )

            courses_for_gpa = selected_courses_df[selected_courses_df['Select']]

            if not courses_for_gpa.empty:
                full_course_data = edited_df[edited_df.index.isin(courses_for_gpa.index)]
                total_credits = pd.to_numeric(full_course_data['Credits']).sum()
                total_quality_points = (pd.to_numeric(full_course_data['Credits']) * pd.to_numeric(full_course_data['Grade_Points']).fillna(0)).sum()
                recent_gpa = total_quality_points / total_credits if total_credits > 0 else 0.0
                st.metric(label="Calculated GPA for Selected Courses", value=f"{recent_gpa:.3f}")
                st.info(f"Total selected credits: **{total_credits:.1f}**")
            else:
                st.metric(label="Calculated GPA for Selected Courses", value="0.000")
                st.info("Total selected credits: **0.0**")
else:
    st.info("Please upload transcript(s) and enter the primary/default university name to begin analysis.")