import pypdf
import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize logger for this module
logger = logging.getLogger(__name__)


class PolicyProcessor:
    def __init__(self, data_path):
        """
        Initialize the processor with data path and text splitting parameters.
        """
        self.data_path = data_path
        # Define chunk size and overlap to preserve context for long lists (e.g., CPT codes)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500,
            separators=["\n\n", "\n", ".", " "]
        )
        logger.info(f"PolicyProcessor initialized with data_path: {data_path}")

    def extract_text_from_pdf(self, file_name):
        """
        Extract raw text content from a single PDF file.
        """
        full_path = os.path.join(self.data_path, file_name)
        text = ""
        try:
            with open(full_path, "rb") as file:
                reader = pypdf.PdfReader(file)
                # Iterate through pages and aggregate text
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                logger.debug(f"Successfully extracted {len(reader.pages)} pages from {file_name}")
        except Exception as e:
            logger.error(f"Error reading PDF file {file_name}: {e}", exc_info=True)
        return text

    def get_chunks(self):
        """
        Process all PDFs in the data directory and return a list of text chunks with metadata.
        """
        all_chunks = []
        # Filter for PDF files only
        try:
            files = [f for f in os.listdir(self.data_path) if f.endswith(".pdf")]
            logger.info(f"Found {len(files)} PDF files in {self.data_path}")
        except Exception as e:
            logger.error(f"Failed to access data directory {self.data_path}: {e}")
            return []

        for file in files:
            raw_text = self.extract_text_from_pdf(file)

            if not raw_text.strip():
                logger.warning(f"File {file} appears to be empty or unreadable. Skipping.")
                continue

            # Split raw text into manageable pieces for embedding
            chunks = self.text_splitter.split_text(raw_text)
            logger.info(f"Processed '{file}': Generated {len(chunks)} chunks.")

            for chunk in chunks:
                # Store text along with its source file name
                all_chunks.append({
                    "text": chunk,
                    "source": file
                })

        logger.info(f"Total chunks generated across all files: {len(all_chunks)}")
        return all_chunks