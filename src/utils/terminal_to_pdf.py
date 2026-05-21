import sys
from fpdf import FPDF
import datetime

import io

class TerminalToPDF:
    """
    Utility to capture all terminal output to a PDF file, while still printing to the terminal.
    Usage:
        buffer = TerminalToPDF('output.pdf')
        buffer.start()
        ... # your code with prints
        buffer.stop()
        buffer.save_pdf() # call this at the end
    """
    def __init__(self, pdf_path="terminal_output.pdf", font_size=10, font_family="Courier"):
        self.pdf_path = pdf_path
        self.font_size = font_size
        self.font_family = font_family
        self._stdout = None
        self._buffer = io.StringIO()
        self._active = False

    def start(self):
        self._stdout = sys.stdout
        sys.stdout = self
        self._active = True

    def stop(self):
        if self._active:
            sys.stdout = self._stdout
            self.content = self._buffer.getvalue()
            self._buffer.close()
            self._active = False

    def write(self, text):
        self._stdout.write(text)
        self._buffer.write(text)

    def flush(self):
        self._stdout.flush()
        self._buffer.flush()

    def save_pdf(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font(self.font_family, size=self.font_size)
        pdf.cell(0, 10, f"Terminal Output - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)
        for line in self.content.splitlines():
            pdf.multi_cell(0, 5, line)
        pdf.output(self.pdf_path)
        print(f"[TerminalToPDF] PDF saved to: {self.pdf_path}")
