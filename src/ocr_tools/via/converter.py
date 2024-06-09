from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image


def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    images = convert_from_path(pdf_path)
    return images


def bytes_to_image(
    pdf_bytes: bytes, dpi: int = 400, num_thread: int = 1
) -> list[Image.Image]:
    images = convert_from_bytes(pdf_bytes, dpi=dpi, thread_count=num_thread)
    return images
