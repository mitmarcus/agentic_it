import pdfplumber
from pathlib import Path
from typing import List, Dict, Any

class PDFParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}.")
    
    def extract_text(self) -> str:
        text = ""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def extract_tables(self) -> List[List[List[str]]]:
        tables = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
        return tables

def parse_pdf(pdf_path: str) -> Dict[str, Any]:
    parser = PDFParser(pdf_path)
    text = parser.extract_text()
    
    return {
        'text': text,
        'tables': parser.extract_tables()
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        print("No path to the PDF file was specified.")
        sys.exit(1)
    
    try:
        print(f"Parsing {pdf_file}.")
        result = parse_pdf(pdf_file)

        pdf_path = Path(pdf_file)
        output = pdf_path.with_suffix('.txt')

        script_dir = Path(__file__).parent
        output_folder = script_dir / "out"
        output_folder.mkdir(exist_ok=True)
        output = output_folder / output.name

        with open(output, 'w', encoding = 'utf-8') as file:
            file.write(result['text'])
        print(f"Text saved to {output}.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Path was wrong.")
    except Exception as e:
        print(f"Unexpected error: {e}")