import cv2
import streamlit as st
import pytesseract
import re
import requests
import numpy as np
from pdf2image import convert_from_bytes
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_text(image):
    return pytesseract.image_to_string(image)

def structure_text(text):
    # More flexible regex to handle various formats like "Test: Value Unit (Range)" or "Test Value Unit (Range)"
    pattern = r"(\w+(?:\s+\w+)*)\s*[:\s]+([\d\.]+)\s*(\w+(?:/\w+)?)\s*\(([\d\.\-–]+)\)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    structured_data = []
    for match in matches:
        test_name, value, unit, range_str = match
        try:
            value = float(value)
            range_min, range_max = map(float, range_str.replace('–', '-').split("-"))
            status = "Normal" if range_min <= value <= range_max else "Abnormal"
            structured_data.append({
                "test": test_name.strip(),
                "value": value,
                "range": range_str,
                "unit": unit,
                "status": status
            })
        except ValueError:
            continue  # Skip malformed entries
    return structured_data

def explain_result(test_data, api_key):
    abnormal_tests = [t for t in test_data if t["status"] == "Abnormal"]
    explanations = {}
    for test in abnormal_tests:
        prompt = f"Explain in simple language what it means if the patient's {test['test']} is {test['value']} {test['unit']}, given the normal range is {test['range']}."
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100
                }
            )
            response.raise_for_status()
            explanations[test["test"]] = response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Request7580Exception:
            explanations[test["test"]] = "Error: Could not generate explanation."
    return explanations

def generate_summary_and_suggestions(test_data, explanations, api_key):
    if not test_data:
        return "No results extracted from the report.", []
    
    abnormal_tests = [t for t in test_data if t["status"] == "Abnormal"]
    if not abnormal_tests:
        return "All test results are within normal ranges.", []
    
    summary_prompt = "Summarize these abnormal results in simple language and suggest follow-up actions: " + "; ".join(
        f"{t['test']} is {t['value']} {t['unit']} (normal: {t['range']})" for t in abnormal_tests
    )
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": summary_prompt}],
                "max_tokens": 200
            }
        )
        response.raise_for_status()
        summary = response.json()["choices"][0]["message"]["content"]
        # Extract suggestions (assuming AI includes them as a list or sentences)
        suggestions = [line.strip() for line in summary.split("\n") if line.strip().startswith("- ")]
        summary = summary.split("\n")[0] if suggestions else summary
        return summary, suggestions
    except requests.exceptions.RequestException:
        return "Error: Could not generate summary.", []

def generate_pdf_report(test_data, explanations, summary, suggestions):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph("Medical Report Analysis", styles['Title']))
    story.append(Spacer(1, 12))
    
    for test in test_data:
        result_text = f"<b>{test['test']}</b>: {test['value']} {test['unit']} (Range: {test['range']}) - {test['status']}"
        story.append(Paragraph(result_text, styles['Normal']))
        if test["test"] in explanations:
            story.append(Paragraph(f"Explanation: {explanations[test['test']]}", styles['Normal']))
        story.append(Spacer(1, 12))
    
    story.append(Paragraph("<b>Summary</b>", styles['Heading2']))
    story.append(Paragraph(summary, styles['Normal']))
    story.append(Spacer(1, 12))
    
    if suggestions:
        story.append(Paragraph("<b>Suggestions</b>", styles['Heading2']))
        for suggestion in suggestions:
            story.append(Paragraph(suggestion, styles['Normal']))
    
    story.append(Spacer(1, 12))
    story.append(Paragraph("Note: This tool is for informational purposes only. Consult a doctor for medical advice.", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

st.title("Medical Report Assistant")
uploaded_file = st.file_uploader("Upload a medical report (JPEG/PNG/PDF)", type=["jpg", "png", "pdf"])

if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            images = convert_from_bytes(uploaded_file.read())
            image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
        else:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        processed_image = preprocess_image(image)
        st.image(processed_image, caption="Processed Image", use_container_width=True)
        
        raw_text = extract_text(processed_image)
        # st.subheader("Extracted Text")
        # st.write(raw_text if raw_text else "No text extracted. Try a clearer image or PDF.")
        
        data = structure_text(raw_text)
        st.subheader("Structured Data")
        st.write(data if data else "No data parsed. The report format may not match the expected pattern.")
        
        api_key = st.secrets["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"]
        explanations = explain_result(data, api_key)
        
        st.subheader("Test Results")
        for test in data:
            st.write(f"**{test['test']}**: {test['value']} {test['unit']} (Range: {test['range']}) - {test['status']}")
            if test["test"] in explanations:
                with st.expander(f"Explanation for {test['test']}"):
                    st.write(explanations[test["test"]])
        
        summary, suggestions = generate_summary_and_suggestions(data, explanations, api_key)
        st.subheader("Summary")
        st.write(summary)
        
        if suggestions:
            st.subheader("Suggested Actions")
            for suggestion in suggestions:
                st.write(suggestion)
        
        pdf_buffer = generate_pdf_report(data, explanations, summary, suggestions)
        st.download_button(
            label="Download Report as PDF",
            data=pdf_buffer,
            file_name="medical_report_summary.pdf",
            mime="application/pdf"
        )
        
        st.write("**Note**: This tool is for informational purposes only. Consult a doctor for medical advice.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")