from pypdf import PdfReader

try:
    reader = PdfReader("MediDiagnose_Anemia Detection.docx (1).pdf")
    print("PDF Content Preview:\n")
    # Read first 5 pages to get structure/TOC
    for i in range(min(5, len(reader.pages))):
        print(f"--- Page {i+1} ---")
        print(reader.pages[i].extract_text())
        print("\n")
except Exception as e:
    print(f"Error reading PDF: {e}")
