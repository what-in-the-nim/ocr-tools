import os
import shutil

import streamlit as st

st.title("Split Images into Batches")
st.write("This page is used to split the images into smaller batches.")

with st.expander("Example", expanded=False):
    st.write("**Input:** Path to input folder containing multiple page images.")
    st.code(
        """
        path/to/input/folder
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
    st.write("**Output:** Path to output folder containing multiple batch folders.")
    st.code(
        """
        # Batch size is 3, start batch index is 10.
        path/to/output/folder
        ├── batch_10
        |   ├── 1_0.png
        |   ├── 1_1.png
        |   └── 1_2.png
        ├── batch_11
        |   ├── 1_3.png
        |   ├── 1_4.png
        |   └── 1_5.png
        ├── ...
        └── batch_x
            ├── 2_3.png
            ├── 2_4.png
            └── 2_5.png
        """
    )

# Create a form to prepare the parameters.
st.subheader("Parameters")

# Create a form to prepare the parameters.
with st.form(key="via_form"):
    # Receive input folder path.
    input_folder_path = st.text_input(
        label="Input Folder Path",
        placeholder="path/to/folder",
        help="Path to the folder containing multiple page images.",
    )
    # Create optional output folder path if user want to provide custom path.
    output_folder_path = st.text_input(
        label="Output Folder Path (Optional)",
        placeholder="path/to/folder_batch",
        help="Path to the folder to save the batch folders.",
    )

    col1, col2, _ = st.columns([1, 1, 2])
    # Receive the number of images per batch.
    batch_size = col1.number_input(
        "Images per batch",
        min_value=10,
        max_value=1000,
        step=10,
        value=100,
        help="The number of images per batch folder.",
    )
    # Receive start batch index
    start_batch_index = col2.number_input(
        "Start Batch Index",
        min_value=1,
        step=1,
        value=1,
        help="The starting index for the batch folder.",
    )
    # Submit button to start the process.`
    submitted = st.form_submit_button("Start")

    if not input_folder_path or not submitted:
        st.stop()

    # Validate the input folder path.
    input_folder_path = input_folder_path.strip()
    output_folder_path = output_folder_path.strip()
    if not os.path.exists(input_folder_path):
        st.error(f"Input folder path does not exist: '{input_folder_path}'.")
        st.stop()

    # Validate items in the input folder.
    ## Check if there are any files in the input folder.
    files = sorted(os.listdir(input_folder_path))
    if len(files) == 0:
        st.error(f"Input folder is empty: '{input_folder_path}'.")
        st.stop()
    ## Check if there are some files that are not [png, jpg, jpeg].
    invalid_files = [
        file for file in files if not file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if invalid_files:
        st.warning(
            f"Invalid files found: {', '.join(invalid_files)}. "
            "Ignoring these files..."
        )
    # Get the valid files.
    files = [file for file in files if file not in invalid_files]

    # Get default output folder path.
    if output_folder_path == "":
        # Create a new folder with the same name as the input folder.
        input_folder_basename = os.path.basename(input_folder_path)
        output_folder_basename = f"{input_folder_basename}_batch"
        output_folder_path = os.path.join(
            os.path.dirname(input_folder_path), output_folder_basename
        )

# Start displaying the progress bar.
status_bar = st.status("Processing...", expanded=True)

# Get the total number of batches.
total_batches, remainder = divmod(len(files), batch_size)
if remainder > 0:
    total_batches += 1

status_bar.write(f"Output folder path: {output_folder_path}")

# Batching the images.
progress_bar = status_bar.progress(0, text="Batching images...")
for batch_idx in range(total_batches):
    progress_bar.progress(
        (batch_idx + 1) / total_batches,
        text=f"Batching ({batch_idx + 1}/{total_batches})...",
    )
    # Create batch folder name.
    batch_folder_name = f"batch_{start_batch_index + batch_idx}"
    # Get path to the batch folder and create the folder.
    batch_folder_path = os.path.join(output_folder_path, batch_folder_name)
    # Check if the folder is existed.
    if os.path.exists(batch_folder_path):
        status_bar.update(label="Error!", state="error", expanded=False)
        st.error(
            f"Batch folder path is existed: '{batch_folder_path}'. Please shift the start batch index or remove the old batch folder."
        )
        st.stop()

    os.makedirs(batch_folder_path, exist_ok=True)

    # Get files to be copied to the batch folder.
    # Get the start and end index for the files.
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size
    files_to_copy = files[start_idx:end_idx]

    # Copy files
    for file in files_to_copy:
        file_path = os.path.join(input_folder_path, file)
        shutil.copy2(file_path, batch_folder_path)

status_bar.write("Finished!")
status_bar.update(label="Finished!", state="complete", expanded=False)
st.success("Batching images completed.")
