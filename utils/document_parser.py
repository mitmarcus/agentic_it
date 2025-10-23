import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import re

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
                if not page_text:
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
        
        for tag in soup.find_all(["head", "script", "style"]):
            tag.decompose()

        # this doesn't give relevant info
        for div_id in ["breadcrumb-section", "footer"]:
            for div in soup.find_all("div", id=div_id):
                div.decompose()

        for br in soup.find_all("br"):
            br.replace_with("\n")

        for li in soup.find_all("li"):
            li.insert_before("- ")
            li.append("\n")

        # split at block level, otherwise it'll break on inline tags too (like italics)
        block_tags = ['p', 'div', 'h1','h2','h3','h4','h5','h6', 'li']
        for tag in soup.find_all(block_tags):
            if not tag.get_text(strip=True).endswith("\n"):
                tag.append("\n")

        text = soup.get_text()
        
        # gets rid of extra whitespace
        lines = (line.strip() for line in text.splitlines())
        text = "\n".join(line for line in lines if line)

        text = re.sub(r'\n+', '\n', text)  # collapse multiple newlines
        
        return text
    
    def extract_tables(self) -> List[List[List[str]]]:
        soup = BeautifulSoup(self.html_source, 'html.parser')
        tables = []
        
        for table in soup.find_all('table'):
            table_data = []
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                row_data = [cell.get_text(strip=True) for cell in cells]
                if row_data:  # only add non-empty rows
                    table_data.append(row_data)
            
            if table_data:
                tables.append(table_data)
        
        return tables

def parse_document(file_path: str) -> Dict[str, Any]:
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
        input_source = Path(sys.argv[1])
    else:
        print("No path to the file or URL was specified.")
        sys.exit(1)

    script_dir = Path(__file__).parent
    output_folder = script_dir / "out"
    output_folder.mkdir(exist_ok=True)
    
    try:
        if input_source.is_dir(): # if given the folder with html pages
            files = list(input_source.glob("*.htm")) + list(input_source.glob("*.html"))
            if not files:
                print("There were no HTML files in the folder.")
                sys.exit(0)

            for file in files:
                result = parse_document(str(file))
                output_name = file.stem + '.txt'
                output = output_folder / output_name

                with open(output, 'w', encoding='utf-8') as out:
                    out.write(result['text'])

        elif input_source.is_file(): # if given the pdf
            result = parse_document(str(input_source))
            output_name = input_source.stem + '.txt'
            output = output_folder / output_name

            with open(output, 'w', encoding='utf-8') as out:
                out.write(result['text'])
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")