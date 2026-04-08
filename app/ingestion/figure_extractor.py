import fitz
import os
import uuid
from typing import List, Dict

class FigureExtractor:
    def __init__(self, output_dir="data/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_figures(self, pdf_path: str, paper_id: str) -> List[Dict]:
        """
        Extract all images from a PDF and save them to disk.
        Returns a list of dicts with metadata about the figures.
        """
        doc = fitz.open(pdf_path)
        figures = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_id = str(uuid.uuid4())
                image_filename = f"{paper_id}_p{page_index+1}_i{img_index}.{image_ext}"
                image_path = os.path.join(self.output_dir, image_filename)

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                figures.append({
                    "figure_id": image_id,
                    "paper_id": paper_id,
                    "page": page_index + 1,
                    "image_path": image_path
                })
        
        doc.close()
        return figures
