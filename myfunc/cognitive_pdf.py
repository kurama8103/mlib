from io import StringIO
from pdfminer.high_level import extract_text_to_fp


def Cognitive_PDF(FILE_PATH):
    output_string = StringIO()
    with open(FILE_PATH, "rb") as fin:
        extract_text_to_fp(fin, output_string)
    return output_string.getvalue()
