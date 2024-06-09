import io
import os
import zipfile

import streamlit as st

from ocr_tools.via import converter

st.title("Convert PDF to Images")
st.write("This page convert each page in `pdf` file into `png` images.")

with st.expander("Example"):
    st.write("**Input:** Multiple `pdf` files")
    st.code("`1.pdf`, `2.pdf`, ...")
    st.write("**Output:** Zip file containing a `png` images of each `pdf` file.")
    st.code(
        """
        images.zip
        ├── 1_0.png
        ├── 1_1.png
        ├── 1_2.png
        ├── ...
        ├── 2_0.png
        ├── 2_1.png
        ├── 2_2.png
        └── ...
        """
    )

# Create a form to prepare the parameters.
st.subheader("Parameters")
with st.form(key="pdf_form"):
    # Receive input files from user.
    pdf_files = st.file_uploader(
        "Upload PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Input file can be single or multiple pdf files.",
    )
    # Receive the output filename.
    zip_filename = st.text_input(
        "Output zip filename",
        value="images.zip",
        placeholder="output.zip",
        help='The output zip filename. Must type in ".zip" extension.',
    )
    # Receive num_thread to speed up the process.
    col1, col2, _ = st.columns([1, 1, 2])
    total_threads = os.cpu_count()
    default_thread = min(4, total_threads)
    num_threads = col1.number_input(
        label="Number of threads",
        min_value=1,
        max_value=total_threads,
        value=default_thread,
        help="Number of threads to speed up the process, may be slower if too many threads.",
    )
    # Receive the DPI for the output images.
    dpi = col2.number_input(
        label="DPI",
        min_value=200,
        max_value=800,
        value=400,
        step=50,
        help="DPI for the output image. Higher DPI will have higher resolution, but slower processing.",
    )
    # Submit button to start the process.`
    submitted = st.form_submit_button("Convert")
    if len(pdf_files) == 0 or not submitted:
        st.stop()

# Start displaying the progress bar.
status_bar = st.status("Converting PDF(s) to images...", expanded=True)

# Create a buffer for zip file.
status_bar.write("Creating zip file...")
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w") as zip_file:
    # Iterate each files
    status_bar.write("Converting images...")
    progress_bar = status_bar.progress(0, text="Processing...")
    total_files = len(pdf_files)
    for file_idx, pdf_file in enumerate(pdf_files):
        progress_bar.progress(
            (file_idx + 1) / total_files,
            text=f"Processing ({file_idx + 1}/{total_files}): {pdf_file.name}...",
        )
        # Open image content.
        pdf_content = pdf_file.getvalue()
        file_name = pdf_file.name
        base_name = file_name.split(".")[0]

        # Crop image into patches.
        images = converter.bytes_to_image(pdf_content, dpi=dpi, num_thread=num_threads)

        # Save each images to directory
        for page_number, image in enumerate(images):
            page_buffer = io.BytesIO()
            image.save(page_buffer, format="png")
            zip_file.writestr(f"{base_name}_{page_number}.png", page_buffer.getvalue())

# Create download button.
status_bar.write("Finished!")
status_bar.update(label="Finished!", state="complete", expanded=False)
st.download_button(
    label="Download images",
    data=zip_buffer,
    file_name=zip_filename,
    mime="application/zip",
)
