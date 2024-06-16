from pathlib import Path

import streamlit as st

from ocr_tools.via import VIAConfig

st.title("VIA Generation")
st.write("This page is used to generate a VIA for VGG Image Annotator.")

# Display the examples.
st.subheader("Examples")
## Example 1: Single Batch Folder
with st.expander("Example 1: Single Batch Folder"):
    st.write("**Input:** path to the folder which contain multiple file pages.")
    st.code(
        """
        path/to/batch_folder
        ├── 1_0.png
        ├── 1_1.png
        └── ...
        """
    )
    st.write("**Output:** VIA `json` file in the same folder.")
    st.code(
        """
        path/to/batch_folder
        ├── 1_0.png
        ├── 1_1.png
        ├── ...
        └── via.json
        """
    )

## Example 2: Multiple Batch Folders
with st.expander("Example 2: Multiple Batch Folders"):
    st.write("**Input:** path to the folder which contain multiple batch folders.")
    st.code(
        """
        path/to/folder
        ├── batch_1
        │   ├── 1_0.png
        │   ├── 1_1.png
        │   └── ...
        ├── batch_2
        │   ├── 2_0.png
        │   └── ...
        └── ...
        """
    )
    st.write("**Output:** VIA `json` file in each folder.")
    st.code(
        """
        path/to/folder
        ├── batch_1
        │   ├── 1_0.png
        │   ├── 1_1.png
        │   ├── ...
        │   └── via.json
        ├── batch_2
        │   ├── 2_0.png
        │   ├── ...
        │   └── via.json
        └── ...
        """
    )


# Create a form to prepare the parameters.
st.subheader("Parameters")

# Create a form to prepare the parameters.
with st.form(key="via_form"):
    mode = st.radio(
        label="Mode",
        options=["Single Batch", "Multiple Batch"],
        index=1,
        horizontal=True,
        help="The mode to generate VIA.",
    )

    path_placeholder = (
        "path/to/batch_folder" if mode == "Single Batch" else "path/to/folder"
    )
    folder_path = st.text_input(
        "Folder Path",
        value="",
        placeholder=path_placeholder,
        help="The path to the folder.",
    )

    overwrite = st.checkbox(
        "Overwrite", value=False, help="Overwrite the existing VIA."
    )

    # with st.expander("Advanced Options"):
    #     st.subheader("Region attributes")
    #     text_types = st_tags(
    #         label="Region Types",
    #         text="Press enter to add more",
    #         value=["text"],
    #         suggestions=[
    #             "five",
    #             "six",
    #             "seven",
    #             "eight",
    #             "nine",
    #             "three",
    #             "eleven",
    #             "ten",
    #             "four",
    #         ],
    #     )
    #     asdf = st.data_editor({"text": {"type": "text", "description": "", "default_value": ""}})
    #     st.write(asdf)
    submit_button = st.form_submit_button(label="Generate VIA")

    if not submit_button or folder_path == "":
        st.stop()

    folder_path = Path(folder_path)
    # Validate if the folder path exists.
    if not folder_path.exists():
        st.error("The folder path does not exist.")
        st.stop()

    # Get all folders to create VIA.
    if mode == "Single Batch":
        folder_to_generate = [folder_path]
    elif mode == "Multiple Batch":
        # All directories in the folder.
        folder_to_generate = [f for f in folder_path.iterdir() if f.is_dir()]
        # Raise error if no folder to generate.
        if len(folder_to_generate) == 0:
            st.error(
                "No folder is found. Please double check the `Folder Path` or the `Mode` is selected correctly."
            )
            st.stop()

    # Validate if json file exists if overwrite is False.
    if not overwrite:
        for folder in folder_to_generate:
            via_path = folder / f"{folder.name}.json"
            if via_path.exists():
                st.error(
                    f"The VIA json file already exists at {via_path}. Please enable `Overwrite` to overwrite the existing VIA or delete the existing VIA."
                )
                st.stop()

# Start displaying the progress bar.
progress_bar = st.status(f"Generating VIA in {mode} mode...", expanded=True)

# Iterate through the folders to generate VIA.
total_folders = len(folder_to_generate)
for folder_idx, folder in enumerate(folder_to_generate):
    # Update the progress bar.
    progress_bar.progress(
        (folder_idx + 1) / total_folders, text=f"Processing {folder.name}..."
    )
    # Create VIA dict.
    via_config = VIAConfig(folder)
    via_config.save()


progress_bar.update(label="Finished!", state="complete", expanded=False)
st.success("VIA generation is completed!")
