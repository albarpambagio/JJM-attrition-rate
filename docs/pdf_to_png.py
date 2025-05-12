import fitz  # PyMuPDF
import os

pdf_path = 'docs/dashboard.pdf'
output_dir = 'docs'

# Open the PDF
doc = fitz.open(pdf_path)

for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=200)
    output_path = os.path.join(output_dir, f'dashboard_page_{page_num+1}.png')
    pix.save(output_path)
    print(f'Saved: {output_path}')

doc.close() 