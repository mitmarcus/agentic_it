import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import re

class PDFParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)

    def _open_pdf(self):
        return pdfplumber.open(self.pdf_path)
    
    def extract_text(self) -> str:
        result = []
        with self._open_pdf() as pdf:
            for page in pdf.pages:
                if not page.chars: # skips blanks
                    continue

                word = []
                prev_style = None
                prev_x = None
                prev_y = None
                
                for char in page.chars:
                    font = char.get('fontname', '').lower()
                    style = ('bold' in font, 'italic' in font or 'oblique' in font)
                    
                    # new word if different line/space/style change
                    new_line = prev_y and abs(char['top'] - prev_y) > 5
                    space = prev_x and (char['x0'] - prev_x) > 3
                    
                    if word and (new_line or space or style != prev_style):
                        result.append(self._format(word, prev_style))
                        result.append('\n' if new_line else ' ')
                        word = []
                    
                    word.append(char['text'])
                    prev_style = style
                    prev_x = char.get('x1')
                    prev_y = char['top']
                
                if word:
                    result.append(self._format(word, prev_style))
                result.append('\n\n')
        
        return ''.join(result)
    
    def _format(self, chars, style):
        text = ''.join(chars)
        bold, italic = style
        if bold and italic:
            return f"***{text}***"
        if bold:
            return f"**{text}**"
        if italic:
            return f"*{text}*"
        return text
    
    def extract_tables(self) -> List[List[List[str]]]:
        tables = []
        with self._open_pdf() as pdf:
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
        with open(html_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return cls(content)
    
    def _table_to_text(self, table) -> str:
        """
        Convert an HTML table to readable text format.
        
        For maintenance schedules, contact lists, etc., this creates
        a format like:
        
        | Category | Action | Services Impacted |
        | Network | Firmware Update | Fortinet sites will have downtime |
        """
        rows = table.find_all('tr')
        if not rows:
            return ""
        
        # Extract all rows
        table_data = []
        for row in rows:
            cells = row.find_all(['td', 'th'])
            row_data = [cell.get_text(strip=True) for cell in cells]
            if any(row_data):  # Skip completely empty rows
                table_data.append(row_data)
        
        if not table_data:
            return ""
        
        # Format as readable text
        # If first row looks like headers (th tags or short cells), treat as headers
        lines = []
        headers = table_data[0] if table_data else []
        
        # Simple table format: "Field: Value" pairs if 2 columns, else pipe-separated
        if len(headers) == 2:
            # Key-value style
            for row in table_data:
                if len(row) >= 2 and row[0] and row[1]:
                    lines.append(f"{row[0]}: {row[1]}")
        else:
            # Multi-column: use headers as context
            for i, row in enumerate(table_data):
                if i == 0:
                    # Header row
                    lines.append(" | ".join(row))
                    lines.append("-" * 40)
                else:
                    # Data row - combine with headers for context
                    row_parts = []
                    for j, cell in enumerate(row):
                        if cell:
                            header = headers[j] if j < len(headers) else ""
                            if header and header != cell:
                                row_parts.append(f"{header}: {cell}")
                            else:
                                row_parts.append(cell)
                    if row_parts:
                        lines.append(" | ".join(row_parts))
        
        return "\n".join(lines)
    
    def get_title(self) -> str:
        soup = BeautifulSoup(self.html_source, 'html.parser')
        title = soup.find('title')
        title = re.sub(r"^Public\s*:\s*", "", title.get_text(strip=True))
        title = re.sub(r'[\\/*?:"<>|]', "_", title) # sanitize for file names
        return title

    
    def extract_text(self) -> str:
        soup = BeautifulSoup(self.html_source, 'html.parser')
        
        # Remove non-content elements
        for tag in soup.find_all(["head", "script", "style", "noscript", "iframe", "svg"]):
            tag.decompose()
        
        # Convert tables to readable text format (prevents fragmentation in chunking)
        for table in soup.find_all('table'):
            table_text = self._table_to_text(table)
            if table_text:
                table.replace_with(BeautifulSoup(f'<div class="table-content">\n{table_text}\n</div>', 'html.parser'))

        # Remove navigation/footer noise
        for div_id in ["breadcrumb-section", "footer", "navigation", "sidebar"]:
            for div in soup.find_all("div", id=div_id):
                div.decompose()
        
        # Remove common footer classes
        for footer in soup.find_all(["footer", "nav"]):
            footer.decompose()
        for div in soup.find_all("div", class_=re.compile(r'footer|page-footer|document-footer', re.I)):
            div.decompose()
        
        # Remove image tags completely (they leak as <img ...> in text)
        for img in soup.find_all("img"):
            img.decompose()
        
        # Remove links that are just icons/images
        for a in soup.find_all("a"):
            if not a.get_text(strip=True):
                a.decompose()

        for br in soup.find_all("br"):
            br.replace_with("\n")

        for li in soup.find_all("li"):
            li.insert_before("- ")
            li.append("\n")

        # convert style tags so they don't get removed when beautifulsoup get_text()s it
        inline_tags = {
            'strong': ('**', '**'),
            'b': ('**', '**'),
            'em': ('*', '*'),
            'i': ('*', '*'),
            'u': ('_', '_'),
        }

        for tag_name, (start_marker, end_marker) in inline_tags.items():
            for tag in soup.find_all(tag_name):
                tag.insert_before(start_marker)
                tag.insert_after(end_marker)
                tag.unwrap()

        # split at block level, otherwise it'll break on inline tags too (like italics)
        block_tags = ['p', 'div', 'h1','h2','h3','h4','h5','h6', 'li']
        for tag in soup.find_all(block_tags):
            if not tag.get_text(strip=True).endswith("\n"):
                tag.append("\n")

        text = soup.get_text()
        
        # Clean up any remaining HTML artifacts
        text = re.sub(r'<[^>]+>', '', text)  # Strip any remaining HTML tags
        
        # gets rid of extra whitespace
        lines = (line.strip() for line in text.splitlines())
        text = "\n".join(line for line in lines if line)

        text = re.sub(r'\n+', '\n', text)  # collapse multiple newlines
        
        # Remove common page footers (Stibo IT contact info pattern)
        text = re.sub(
            r'\n?\d*\s*Stibo IT\s*\d*\s*\+45\s*[\d\s]+\s*ithelpdesk@stibo\.com\s*',
            '\n',
            text,
            flags=re.IGNORECASE
        )
        
        # Remove page numbers
        text = re.sub(r'^\d{1,3}$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
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
        title = path.stem
    elif extension in ['.html', '.htm']:
        parser = HTMLParser.from_file(file_path)
        try:
            title = parser.get_title()
        except Exception:
            title = path.stem
    else:
        raise ValueError(f"Unsupported file type: {extension}. Supported types: .pdf, .html, .htm")
    
    return {
        'title': title,
        'text': parser.extract_text(),
        'tables': parser.extract_tables(),
        'file_type': extension
    }

# ex. python document_parser.py "path/to/document.html"
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
                output_name = result['title'] + '.txt'
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