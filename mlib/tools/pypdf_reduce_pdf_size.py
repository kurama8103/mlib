# %%
import os
import subprocess
import sys

from pypdf import PdfReader, PdfWriter


def pypdf_reduce_pdf_size(filename_pdf: str, filename_out: str):
    reader = PdfReader(filename_pdf)
    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    for page in writer.pages:
        # ⚠️ This has to be done on the writer, not the reader!
        page.compress_content_streams(level=9)  # This is CPU intensive!

    with open(filename_out, "wb") as f:
        writer.write(f)

    print("before(kb):", os.path.getsize(filename_pdf) // 1000)
    print("after(kb) :", os.path.getsize(filename_out) // 1000)


if __name__ == "__main__":
    if "ipykernel_launcher.py" in sys.argv[0]:
        filename_pdf = "../data/test.pdf"
    else:
        filename_pdf = sys.argv[1]
    filename_out = filename_pdf.replace(".pdf", "_.pdf")
    pypdf_reduce_pdf_size(filename_pdf, filename_out)
    subprocess.call(["open", filename_out])
# %%
