import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
import requests
from bs4 import BeautifulSoup
from openai import OpenAI  # For OpenRouter
from PIL import Image
import pytesseract
import io

# --- APP CONFIGURATION & ONE-TIME SETUP ---
st.set_page_config(layout="wide", page_title="U of A Nursing Transcript Analyzer", page_icon="⚕️")

# --- CONSTANTS & DATABASES ---
UA_PREREQS = {
    "Human Anatomy & Physiology I w/ Lab": {"ua_course": "PSIO 201", "credits": 4, "description": "Integrated study of the structure and function of the human body. Includes cells, tissues, and the integumentary, skeletal, muscular, and nervous systems. Laboratory component required."},
    "Human Anatomy & Physiology II w/ Lab": {"ua_course": "PSIO 202", "credits": 4, "description": "Continuation of PSIO 201. Covers the endocrine, cardiovascular, lymphatic, respiratory, digestive, urinary, and reproductive systems. Laboratory component required."},
    "Microbiology w/ Lab": {"ua_course": "MIC 205A/L", "credits": 4, "description": "Fundamentals of microbiology, including microbial cell structure and function, metabolism, genetics, and the role of microorganisms in disease and the environment. Laboratory component required."},
    "Chemistry I w/ Lab": {"ua_course": "CHEM 151", "credits": 4, "description": "Introduction to the basic principles of chemistry, including atomic structure, chemical bonding, stoichiometry, states of matter, and solutions. Laboratory component required."},
    "Chemistry II w/ Lab": {"ua_course": "CHEM 152", "credits": 4, "description": "Continuation of Chemistry I. Covers chemical kinetics, thermodynamics, equilibrium, acid-base chemistry, and electrochemistry. Laboratory component required."},
    "Statistics": {"ua_course": "MATH 163 or higher", "credits": 3, "description": "Introduction to statistical reasoning, including data collection, descriptive statistics, probability, and inference. Topics include hypothesis testing and confidence intervals."},
    "Lifespan Development": {"ua_course": "FSHD 117", "credits": 3, "description": "Survey of human development from conception through death, covering physical, cognitive, social, and emotional domains across the lifespan."},
    "Nutrition": {"ua_course": "NSC 170C1", "credits": 3, "description": "Basic concepts of human nutrition. Nutrients, their food sources and functions in the body. Interrelationships and application to food selection and health."}
}
GRADE_TO_POINTS = {'A+': 4.0, 'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7, 'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0, 'F': 0.0, 'P': 0.0, 'TP': 0.0, 'Y': 0.0, 'W': 0.0, 'T': 0.0, 'CR': 0.0}

# --- HELPER FUNCTIONS ---

def extract_text_from_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        try:
            with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc: text = "".join(page.get_text() for page in doc)
            return text
        except Exception as e: st.error(f"Error processing PDF {uploaded_file.name}: {e}"); return ""
    elif file_extension in ['png', 'jpg', 'jpeg']:
        try:
            image = Image.open(uploaded_file); text = pytesseract.image_to_string(image)
            return text
        except Exception as e: st.error(f"Error processing image {uploaded_file.name}: {e}"); return ""
    else: st.warning(f"Unsupported file type: {uploaded_file.name}."); return ""

def detect_university_from_text(text):
    text_header = text[:500]
    pattern = re.compile(r'^(.*(University|College|Institute|Community College).*)$', re.MULTILINE | re.IGNORECASE)
    matches = pattern.findall(text_header)
    return matches[0][0].strip() if matches else "Unknown - Please Edit"

# --- SPECIALIZED PARSER SYSTEM ---
def parse_pima_format(text):
    pattern = re.compile(r'^([A-Z]{2,4}\s+\d{3,4}[A-Z]*)\s+(.*?)\s+(\d{1,2}\.\d{2})\s+([A-Z]{1,2})\s+\d{1,3}\.\d{2}(?:\s+[A-Z])?$', re.MULTILINE)
    matches = pattern.findall(text)
    if not matches: return pd.DataFrame()
    return pd.DataFrame(matches, columns=['Course_Code', 'Title', 'Credits', 'Grade'])

def parse_maricopa_format(text):
    pattern = re.compile(r'^([A-Z]{3})\s+(\d{3})\s+(.*?)\s+(\d{1,2}\.\d{2})\s+([A-Z])$', re.MULTILINE)
    matches = pattern.findall(text)
    if not matches: return pd.DataFrame()
    df = pd.DataFrame(matches, columns=['Subject', 'Number', 'Title', 'Credits', 'Grade'])
    df['Course_Code'] = df['Subject'] + ' ' + df['Number']
    return df[['Course_Code', 'Title', 'Credits', 'Grade']]

def parse_nau_transfer_format(text):
    all_courses = []
    course_blocks = text.split("Incoming Course")
    if len(course_blocks) < 2: return pd.DataFrame()
    pattern = re.compile(r'^\s*([A-Z]{2,4})\s+(\d{3,4}[A-Z]*)\s+(.*?)\s+(\d{1,2}\.\d{2})\s+([A-Z][+-]?)', re.MULTILINE)
    for block in course_blocks[1:]:
        match = pattern.search(block)
        if match:
            subject, number, title, credits, grade = match.groups()
            all_courses.append([f"{subject} {number}", title.strip(), credits, grade])
    if not all_courses: return pd.DataFrame()
    return pd.DataFrame(all_courses, columns=['Course_Code', 'Title', 'Credits', 'Grade'])

def parse_courses_from_text(text):
    parsers = [parse_pima_format, parse_maricopa_format, parse_nau_transfer_format]
    for parser_func in parsers:
        df = parser_func(text)
        if not df.empty:
            df['Credits'] = pd.to_numeric(df['Credits'])
            df['Grade_Points'] = df['Grade'].map(GRADE_TO_POINTS).fillna(0)
            st.success(f"Successfully parsed {len(df)} courses using the `{parser_func.__name__}` strategy.", icon="✅")
            return df
    return pd.DataFrame()

# Web search and AI comparison functions remain the same
@st.cache_data(show_spinner="Searching for course description...")
def get_course_description_from_web(course_row):
    university = course_row['University']; course_code = course_row['Course_Code']; course_title = course_row['Title']
    query = f'"{university}" "{course_code}" "{course_title}" course description'
    search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
    fallback_message = f"Auto-search couldn't find a description. For the most accurate comparison, [click to search Google]({search_url}) and paste the result below."
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'}
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status(); soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.find_all('div', class_='g')
        if not search_results: search_results = soup.find_all('div', class_='tF2Cxc')
        if not search_results:
            if "Our systems have detected unusual traffic" in response.text: return f"Auto-search was blocked by Google (CAPTCHA). [Please click here to run the search manually]({search_url}) and paste the result."
            return fallback_message
        page_url = None
        for result in search_results:
            link_tag = result.find('a')
            if link_tag and link_tag.has_attr('href'): page_url = link_tag['href']; break
        if not page_url: return fallback_message
        page_response = requests.get(page_url, headers=headers, timeout=10)
        page_response.raise_for_status(); page_soup = BeautifulSoup(page_response.text, 'html.parser'); page_text = page_soup.get_text()
        match_pos = page_text.lower().find(course_code.lower())
        if match_pos == -1: match_pos = page_text.lower().replace(" ", "").find(course_code.lower().replace(" ",""))
        if match_pos != -1:
            start_index = max(0, match_pos - 100); end_index = start_index + 600
            description = ' '.join(page_text[start_index:end_index].split())
            return description + "..."
        return f"Found a page, but couldn't auto-extract. [Try this manual search link]({search_url}) and paste the result."
    except Exception: return fallback_message

@st.cache_data(show_spinner="AI Assistant is comparing courses...")
def compare_courses_with_openrouter(ua_context, student_context):
    if "OPENROUTER_API_KEY" not in st.secrets:
        return {"decision": "Error", "score": 0, "reason": "OpenRouter API key not found in Streamlit Secrets."}
    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
        prompt = f"""
        You are an expert university admissions advisor's assistant providing expert analysis on course equivalency. Your task is to review a transfer credit request.
        UofA_COURSE: "{ua_context}"
        APPLICANT_COURSE: "{student_context}"
        Based on all provided information, determine if the applicant's course is an acceptable equivalent.
        Respond in a specific format: a numerical percentage match score from 0 to 100, a one-word decision ('Yes', 'No', or 'Maybe'), and a brief one-sentence reason.
        Example 1: Score: 95%, Decision: Yes, Reason: The course titles and descriptions cover nearly identical topics...
        Example 2: Score: 40%, Decision: No, Reason: The applicant's course is a general survey...
        Now, analyze the courses.
        """
        response = client.chat.completions.create(model="google/gemma-7b-it", messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=150)
        ai_response_text = response.choices[0].message.content
        score_match = re.search(r"Score: (\d+)%", ai_response_text)
        decision_match = re.search(r"Decision: (Yes|No|Maybe)", ai_response_text, re.IGNORECASE)
        reason_match = re.search(r"Reason: (.*)", ai_response_text, re.IGNORECASE)
        result = {"score": int(score_match.group(1)) if score_match else 0, "decision": decision_match.group(1).capitalize() if decision_match else "Unknown", "reason": reason_match.group(1).strip() if reason_match else "Could not parse AI response."}
        return result
    except Exception as e:
        return {"decision": "Error", "score": 0, "reason": f"An API error occurred: {e}"}

# --- MAIN APP INTERFACE ---
st.title("⚕️ U of A MEPN Transcript Analyzer")
st.markdown("An internal tool for admissions advisors to streamline prerequisite and GPA evaluation.")

if 'prereq_matches' not in st.session_state: st.session_state.prereq_matches = {}
with st.sidebar:
    st.header("Controls")
    uploaded_files = st.file_uploader("1. Upload Transcript(s) (PDF, PNG, JPG)", type=['pdf', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)
    if st.button("Reset Analysis"): st.session_state.prereq_matches = {}; st.rerun()

if uploaded_files:
    all_courses_list = []
    failed_files = []

    with st.spinner("Processing uploaded transcripts..."):
        for uploaded_file in uploaded_files:
            raw_text = extract_text_from_file(uploaded_file)
            
            # --- FIX: Check for OCR failure on a per-file basis ---
            if not raw_text.strip():
                failed_files.append(uploaded_file.name)
                continue # Skip to the next file
            
            detected_university = detect_university_from_text(raw_text)
            parsed_df = parse_courses_from_text(raw_text)
            
            if not parsed_df.empty:
                parsed_df['Source'] = uploaded_file.name
                parsed_df['University'] = detected_university
                all_courses_list.append(parsed_df)
            else:
                # This file had text but no parser could understand it
                failed_files.append(uploaded_file.name)

    # --- FIX: Display a consolidated, helpful error message for all failed files ---
    if failed_files:
        st.error(
            f"**Automated parsing failed for the following file(s):** `{', '.join(failed_files)}`\n\n"
            "**Common Causes:**\n"
            "- **Watermarks or low image quality** can confuse the OCR.\n"
            "- The transcript may be in a **new or unsupported format**.\n\n"
            "**Solution:** Please review the raw text below and use the **manual data entry** feature to add any missing courses from these files.",
            icon="⚠️"
        )

    # Display raw text in an expander for debugging purposes
    with st.expander("Show Raw OCR Text from Transcripts"):
        for uploaded_file in uploaded_files:
            st.subheader(f"OCR Text for: {uploaded_file.name}")
            # Re-extracting for display to avoid storing all text in memory
            raw_text_display = extract_text_from_file(uploaded_file)
            st.text_area(f"text_{uploaded_file.name}", raw_text_display, height=200, disabled=True)
    
    # Logic to handle the data editor, whether data was parsed or not
    if not all_courses_list:
        st.subheader("Manual Course Entry")
        st.info("Since no courses were automatically parsed, please add them below.")
        edited_df = st.data_editor(pd.DataFrame(columns=['University', 'Source', 'Course_Code', 'Title', 'Credits', 'Grade', 'Grade_Points']), num_rows="dynamic", use_container_width=True)
    else:
        master_df = pd.concat(all_courses_list, ignore_index=True)
        st.subheader("Parsed & Editable Transcript Data")
        st.info("The table below shows all successfully parsed courses. You can edit any cell or add rows for courses from failed files.")
        edited_df = st.data_editor(master_df, num_rows="dynamic", use_container_width=True,
            column_config={ "University": st.column_config.TextColumn("University", help="Auto-detected, but editable."), "Grade_Points": st.column_config.NumberColumn("Grade Points", format="%.1f"), "Source": st.column_config.TextColumn("Source File") },
            column_order=['University', 'Source', 'Course_Code', 'Title', 'Credits', 'Grade', 'Grade_Points'], key="course_editor")
    
    st.divider()
    
    # The rest of the UI tabs remain the same and will function correctly
    tab1, tab2 = st.tabs(["➡️ Prerequisite Analysis Workflow", "📊 Final Summary Report"])
    with tab1:
        # Prerequisite matching logic (unchanged)
        st.header("Prerequisite Matching")
        st.markdown("For each U of A requirement, select the applicant's course that may fulfill it.")
        if edited_df.empty:
            st.warning("No courses to match. Please add courses in the table above.")
        else:
            unique_courses_df = edited_df.drop_duplicates(subset=['Course_Code'])
            course_code_to_title = pd.Series(unique_courses_df.Title.values, index=unique_courses_df.Course_Code).to_dict()
            for req_name, req_details in UA_PREREQS.items():
                with st.expander(f"**{req_name}** ({req_details['ua_course']})", expanded=False):
                    select_key = f"select_{req_details['ua_course']}"
                    current_match_code = st.session_state.prereq_matches.get(req_name, {}).get('Course_Code')
                    options = ['- Not Selected -'] + list(course_code_to_title.keys())
                    try: current_index = options.index(current_match_code) if current_match_code else 0
                    except ValueError: current_index = 0
                    selected_course_code = st.selectbox("Select Applicant's Course", options, index=current_index, key=select_key, format_func=lambda code: f"{code} - {course_code_to_title.get(code, 'Unknown')}" if code != '- Not Selected -' else '- Not Selected -')
                    if selected_course_code != '- Not Selected -':
                        student_course = edited_df[edited_df['Course_Code'] == selected_course_code].iloc[0]
                        st.write(f"**Selected:** {student_course['University']} - {student_course['Course_Code']} | **Credits:** {student_course['Credits']} | **Grade:** {student_course['Grade']}")
                        col1, col2 = st.columns(2)
                        ua_context = f"University of Arizona {req_details['ua_course']} {req_name}. {req_details['description']}"
                        with col1: st.info(f"**U of A Prerequisite Context**"); st.write(ua_context)
                        with col2:
                            st.warning(f"**Applicant's Course Context**")
                            base_student_context = f"{student_course['University']} {student_course['Course_Code']} {student_course['Title']}"
                            st.write(base_student_context)
                            scraped_desc = get_course_description_from_web(student_course)
                            description_to_show = ""
                            if "Auto-search" in scraped_desc or "couldn't auto-extract" in scraped_desc: st.info(scraped_desc, icon="💡")
                            else: description_to_show = scraped_desc
                            manual_desc = st.text_area("Verify or paste description here:", value=description_to_show, key=f"desc_{req_details['ua_course']}", height=150)
                            student_context_full = f"{base_student_context}. {manual_desc}"
                        analysis_result = compare_courses_with_openrouter(ua_context, student_context_full)
                        st.progress(analysis_result['score'] / 100, text=f"AI Match Score: {analysis_result['score']}%")
                        decision = analysis_result['decision']
                        if decision == "Yes": st.success(f"**AI Decision: {decision}** - {analysis_result['reason']}")
                        elif decision == "No": st.error(f"**AI Decision: {decision}** - {analysis_result['reason']}")
                        else: st.warning(f"**AI Decision: {decision}** - {analysis_result['reason']}")
                        if st.button("Confirm this Match", key=f"confirm_{req_details['ua_course']}"):
                            st.session_state.prereq_matches[req_name] = student_course.to_dict(); st.toast(f"Match for {req_name} confirmed!", icon="✅"); st.rerun()

    with tab2:
        # Summary report and GPA calculation logic (unchanged)
        st.header("Evaluation Summary")
        st.subheader("Confirmed Prerequisite Matches")
        if not st.session_state.prereq_matches: st.warning("No prerequisite matches have been confirmed yet.")
        else:
            summary_data = []
            for req_name, req_details in UA_PREREQS.items():
                match = st.session_state.prereq_matches.get(req_name)
                summary_data.append({ "U of A Requirement": req_name, "Status": "✅ Met" if match else "❌ Missing", "Applicant Course": f"{match.get('University', '')} - {match.get('Course_Code', '')}" if match else "---", "Grade": match.get('Grade', '---'), "Credits": match.get('Credits', '---') })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        st.divider(); st.subheader("GPA Calculations"); col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Prerequisite GPA**")
            if st.session_state.prereq_matches:
                matched_courses_df = pd.DataFrame(list(st.session_state.prereq_matches.values()))
                if not matched_courses_df.empty and 'Grade_Points' in matched_courses_df.columns and pd.to_numeric(matched_courses_df['Credits']).sum() > 0:
                    total_credits = pd.to_numeric(matched_courses_df['Credits']).sum()
                    total_quality_points = (pd.to_numeric(matched_courses_df['Credits']) * pd.to_numeric(matched_courses_df['Grade_Points'])).sum()
                    prereq_gpa = total_quality_points / total_credits
                    st.metric(label="Calculated Prereq GPA", value=f"{prereq_gpa:.3f}")
                    if prereq_gpa < 3.0: st.error("GPA is below the typical 3.0 minimum.")
                    else: st.success("GPA meets the typical 3.0 minimum.")
                else: st.metric(label="Calculated Prereq GPA", value="N/A")
            else: st.metric(label="Calculated Prereq GPA", value="N/A")
        with col2:
            st.markdown("**Recent Credits GPA (e.g., Last 60)**")
            st.info("Select the courses to include in the GPA calculation.")
            if edited_df.empty:
                st.warning("No courses to select.")
            else:
                selection_df = edited_df.copy(); selection_df['Select'] = False
                selection_df = selection_df[['Select', 'University', 'Source', 'Course_Code', 'Credits', 'Grade']]
                selected_courses_df = st.data_editor(selection_df, use_container_width=True, hide_index=True, disabled=['University', 'Source', 'Course_Code', 'Credits', 'Grade'])
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
    st.info("Please upload transcript(s) to begin analysis.")