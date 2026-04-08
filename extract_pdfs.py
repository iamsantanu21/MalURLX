import sys
try:
    import pypdf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
    import pypdf

def emit(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ''
        for i in range(min(15, len(reader.pages))):
            text += reader.pages[i].extract_text() + '\n'
        print(f'--- TEXT FOR {pdf_path} ---')
        print(text[:4000])
    except Exception as e:
        print(f'Error reading {pdf_path}: {e}')

pdfs = [
    'NLP_final_0_1 (2) (1).pdf',
    's11416-025-00578-w.pdf',
    'Real-Time-Phishing-_URL_Detection_with_a_Deep_Learning-Based_Browser_Extension.pdf',
    'Malicious_Address_Identifier_MAI_A_Browser_Extension_to_Identify_Malicious_URLs.pdf',
    'A_Holistic_Review_on_Detection_of_Malicious_Browser_Extensions_and_Links_using_Deep_Learning.pdf'
]

for p in pdfs:
    emit(p)
