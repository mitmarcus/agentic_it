import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup

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

class HTMLParser:
    def __init__(self, html_source: str):
        self.html_source = html_source
        if not self.html_source:
            raise FileNotFoundError("HTML source not found.")
        
    @classmethod
    def from_file(cls, html_path: str):
        """Create parser from local HTML file"""
        html_file = Path(html_path)
        if not html_file.exists():
            raise FileNotFoundError(f"HTML file not found at {html_path}.")
        
        with open(html_file, 'r', encoding='utf-8') as file:
            content = file.read()
        return cls(content)
    
    def extract_text(self) -> str:
        """Extract clean text from HTML"""
        soup = BeautifulSoup(self.html_source, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text and clean it up
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_tables(self) -> List[List[List[str]]]:
        """Extract tables from HTML"""
        soup = BeautifulSoup(self.html_source, 'html.parser')
        tables = []
        
        for table in soup.find_all('table'):
            table_data = []
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                row_data = [cell.get_text(strip=True) for cell in cells]
                if row_data:  # Only add non-empty rows
                    table_data.append(row_data)
            
            if table_data:
                tables.append(table_data)
        
        return tables

def parse_document(file_path: str) -> Dict[str, Any]:
    """Parse either PDF or HTML file based on extension"""
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if extension == '.pdf':
        parser = PDFParser(file_path)
    elif extension in ['.html', '.htm']:
        parser = HTMLParser.from_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}. Supported types: .pdf, .html, .htm")
    
    return {
        'text': parser.extract_text(),
        'tables': parser.extract_tables(),
        'file_type': extension
    }

# ex. document_parser.py "path/to/document.pdf"
# ex. document_parser.py "path/to/document.html"
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_source = sys.argv[1]
    else:
        print("No path to the file or URL was specified.")
        sys.exit(1)
    
    try:
        print(f"Parsing file: {input_source}")
        result = parse_document(input_source)
        input_path = Path(input_source)
        output_name = input_path.stem + '.txt'

        # Create output directory and file
        script_dir = Path(__file__).parent
        output_folder = script_dir / "out"
        output_folder.mkdir(exist_ok=True)
        output = output_folder / output_name

        with open(output, 'w', encoding='utf-8') as file:
            file.write(result['text'])
        
        print(f"Text saved to {output}.")
        print(f"Found {len(result['tables'])} tables.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")