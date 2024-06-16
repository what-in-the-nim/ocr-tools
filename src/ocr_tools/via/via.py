import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

VIA_TEMPLATE = {
    "_via_settings": {
        "ui": {
            "annotation_editor_height": 25,
            "annotation_editor_fontsize": 0.8,
            "leftsidebar_width": 18,
            "image_grid": {
                "img_height": 80,
                "rshape_fill": "none",
                "rshape_fill_opacity": 0.3,
                "rshape_stroke": "yellow",
                "rshape_stroke_width": 2,
                "show_region_shape": True,
                "show_image_policy": "all",
            },
            "image": {
                "region_label": "text",
                "region_color": "__via_default_region_color__",
                "region_label_font": "10px Sans",
                "on_image_annotation_editor_placement": "NEAR_REGION",
            },
        },
        "core": {"buffer_size": 18, "filepath": {}, "default_filepath": ""},
        "project": {"name": "batch_1"},
    },
    "_via_img_metadata": {
        "Note_1-001.png1209689": {
            "filename": "Note_1-001.png",
            "size": 1209689,
            "regions": [],
            "file_attributes": {},
        },
    },
    "_via_attributes": {
        "region": {"text": {"type": "text", "description": "", "default_value": ""}},
        "file": {},
    },
    "_via_data_format_version": "2.0.10",
    "_via_image_id_list": [],
}


@dataclass
class VIAConfig:

    project_dir: str | Path
    config: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.project_dir = (
            Path(self.project_dir)
            if isinstance(self.project_dir, str)
            else self.project_dir
        )
        self.config = self.create_via_dict(self.project_dir)

    @staticmethod
    def create_via_dict(project_dir: str) -> dict[str, Any]:
        """Create VIA json file from the project directory."""
        project_dir: Path = Path(project_dir)

        # Create content in VIA json file
        _via_image_id_list = list()
        _via_img_metadata = dict()

        # Iterate through the files in the project directory.
        for filename in project_dir.iterdir():
            # Skip if the file is not a image file.
            if filename.suffix not in {".png", ".jpg", ".jpeg"}:
                continue

            # Create via_id from filename and file size.
            file_size = os.stat(filename).st_size
            via_id = f"{filename.name}{file_size}"  ## e.g. image_001.png1209689

            # Create metadata for the image.
            metadata = {
                "filename": filename.name,
                "size": file_size,
                "regions": [],
                "file_attributes": {},
            }

            # Append via_id and metadata to the VIA json file.
            _via_image_id_list.append(via_id)
            _via_img_metadata[via_id] = metadata

        # Create VIA json file.
        via_dict = VIA_TEMPLATE.copy()
        via_dict["_via_image_id_list"] = _via_image_id_list
        via_dict["_via_img_metadata"] = _via_img_metadata
        via_dict["_via_settings"]["project"]["name"] = project_dir.name

        return via_dict

    def save(
        self, output_dir: Optional[str | Path] = None, output_filename: str = None
    ) -> None:
        """Save the VIA json file to the save path."""
        if output_dir is not None:
            # Convert output_dir to Path if it is a string.
            output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        else:
            # Use the project directory as the output directory.
            output_dir = self.project_dir
        # Create output_dir if it does not exist.
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create output_name if it is None.
        output_filename = (
            f"{self.project_dir.name}.json"
            if output_filename is None
            else output_filename
        )
        save_path = output_dir / output_filename
        # Save the VIA json file.
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
