import logging
from pathlib import Path
from typing import Dict, Optional, List, Union
import PyPDF2
from docx import Document
import os

class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_text_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Load text content with encoding fallback"""
        encodings = ['utf-8', 'latin-1', 'utf-16', 'ascii']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    if content.strip():
                        return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Error loading {file_path} with {encoding}: {e}")
        return None

    def load_pdf_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Extract text from PDF with PyPDF2"""
        try:
            text = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    if text_content := page.extract_text():
                        text.append(text_content)
            return '\n'.join(text) if text else None
        except Exception as e:
            self.logger.error(f"PDF load failed: {file_path} - {str(e)}")
            return None

    def load_docx_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Extract text from DOCX files"""
        try:
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return '\n'.join(paragraphs) if paragraphs else None
        except Exception as e:
            self.logger.error(f"DOCX load failed: {file_path} - {str(e)}")
            return None

    def load_file(self, file_path: Union[str, Path]) -> Optional[Dict[str, str]]:
        """Universal file loader with metadata and validation"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if path.stat().st_size == 0:
                raise ValueError("File is empty")
            if not os.access(file_path, os.R_OK):
                raise PermissionError("No read permission")

            loader_map = {
                '.txt': self.load_text_file,
                '.pdf': self.load_pdf_file,
                '.docx': self.load_docx_file
            }

            suffix = path.suffix.lower()
            if suffix not in loader_map:
                raise ValueError(f"Unsupported file type: {suffix}")

            if content := loader_map[suffix](path):
                return {
                    'path': str(path),
                    'filename': path.name,
                    'text': content,
                    'type': suffix[1:],
                    'size': path.stat().st_size
                }
            raise ValueError("Failed to extract content")
            
        except Exception as e:
            self.logger.error(f"File load error: {path} - {str(e)}")
            raise

    def load_directory(self, dir_path: Union[str, Path]) -> List[Dict[str, str]]:
        """Batch load documents from directory with validation"""
        try:
            path = Path(dir_path)
            if not path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")
            if not os.access(dir_path, os.R_OK):
                raise PermissionError("No read permission")

            documents = []
            for file_path in path.glob('*'):
                try:
                    if doc := self.load_file(file_path):
                        documents.append(doc)
                except Exception as e:
                    self.logger.warning(f"Skipped {file_path}: {str(e)}")
                    continue
            
            if not documents:
                raise ValueError("No supported documents found")
            return documents
            
        except Exception as e:
            self.logger.error(f"Directory load error: {path} - {str(e)}")
            raise