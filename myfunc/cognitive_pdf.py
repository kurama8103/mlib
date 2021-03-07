
def Cognitive_PDF(PATH_PDF):
    from io import StringIO
    from pdfminer.high_level import extract_text_to_fp

    output_string = StringIO()
    with open(PATH_PDF, 'rb') as fin:
        extract_text_to_fp(fin, output_string)
    return output_string.getvalue()
