from pathlib import Path

import streamlit as st

st.title("Tool Page")
st.write("This page is used to get insights about the project.")

# Provide examples of the project directory.
expander = st.expander("**Project Structure**")
expander.write("Can be anything, it will find all related files.")
expander.code(
    """
    path/to/project
    ├── 1.pdf
    ├── 2.pdf
    ├── 3.pdf
    ├── ...
    ├── 1.json
    ├── 2.json
    └── ...
    """
)

# Create a form to prepare the parameters.
st.subheader("Project Name")
with st.form(key="tool_form"):
    # Receive the project directory.
    project_dir = st.text_input(
        "Project Directory",
        value="",
        placeholder="path/to/project",
        help="The project directory.",
    )
    # Submit button to start the process.`
    submitted = st.form_submit_button("Inspect")

    if not submitted:
        st.stop()

    project_dir = Path(project_dir)
    # Validate the project directory.
    if not project_dir.exists():
        st.error("Project directory does not exist.")
        st.stop()

# Find total number of pdf files in the project directory recursively.
pdf_files = list(project_dir.glob("**/*.pdf"))
total_pdf_files = len(pdf_files)
st.write(f"Total PDF files: {total_pdf_files}")
with st.expander("PDF Files"):
    for pdf_file in pdf_files:
        st.write(pdf_file)

# Find total json files in the project directory recursively.
json_files = list(project_dir.glob("**/*.json"))
total_json_files = len(json_files)
st.write(f"Total JSON files: {total_json_files}")
with st.expander("JSON Files"):
    for json_file in json_files:
        st.write(json_file)
