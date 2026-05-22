import os
from fpdf import FPDF
from PIL import Image

class CityscapesReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Cityscapes Analysis Report', ln=True, align='C')
        self.ln(5)

    def add_section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, title, ln=True)
        self.ln(2)

    def add_text(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_image(self, image_path, w=180):
        if os.path.exists(image_path):
            self.image(image_path, w=w)
            self.ln(4)
        else:
            self.add_text(f"[Image not found: {image_path}]")

def generate_cityscapes_report(output_pdf='cityscapes_report.pdf'):
    pdf = CityscapesReportPDF()
    pdf.add_page()

    # 1. Total number of images
    try:
        with open('outputs/statistics/stats_summary.json', 'r') as jf:
            stats = __import__('json').load(jf)
    except Exception as e:
        stats = None
        pdf.add_text(f"[Could not load outputs/statistics/stats_summary.json: {e}]")

    pdf.add_section_title('1. Total number of images')
    if stats and 'resolutions' in stats:
        pdf.add_text(f"Total images: {len(stats['resolutions'])}")
    else:
        pdf.add_text('Total images: [Data not available]')

    # 2. Original image resolutions
    pdf.add_section_title('2. Original image resolutions')
    
    if stats and 'resolutions' in stats:
        unique_res = set((r['width'], r['height']) for r in stats['resolutions'])
        pdf.add_text(f"Unique resolutions: {sorted(unique_res)}")
    else:
        pdf.add_text('Unique resolutions: [Data not available]')
    pdf.add_image('outputs/plots/resolution_histogram.png')

    # 3. List of present classes
    pdf.add_section_title('3. List of present classes')
    
    if stats and 'classes' in stats:
        class_names = [c['class_name'] for c in stats['classes']]
        pdf.add_text(f"Classes present ({len(class_names)}): {', '.join(class_names)}")
    else:
        pdf.add_text('Classes present: [Data not available]')

    # 4. Absolute and relative pixel frequency per class
    pdf.add_section_title('4. Absolute and relative pixel frequency per class')
    
    if stats and 'classes' in stats:
        table = 'Class | Pixels | % Pixels\n---|---|---\n'
        for c in stats['classes']:
            table += f"{c['class_name']} | {c['pixel_count']} | {c['pixel_percent']:.2f}\n"
        pdf.add_text(table)
    else:
        pdf.add_text('[Data not available]')
    pdf.add_image('outputs/plots/pixel_frequency_per_class.png')

    # 5. Frequency of appearance of each class per image
    pdf.add_section_title('5. Frequency of appearance of each class per image')
    
    if stats and 'classes' in stats:
        table = 'Class | Appears in Images\n---|---\n'
        for c in stats['classes']:
            table += f"{c['class_name']} | {c['appears_in_images']}\n"
        pdf.add_text(table)
    else:
        pdf.add_text('[Data not available]')
    pdf.add_image('outputs/plots/class_appearance_per_image.png')

    # 6. Degree of class imbalance (max/min)
    pdf.add_section_title('6. Degree of class imbalance')
    
    if stats and 'classes' in stats:
        pixel_counts = [c['pixel_count'] for c in stats['classes'] if c['pixel_count'] > 0]
        if pixel_counts:
            imbalance = max(pixel_counts) / min(pixel_counts) if min(pixel_counts) > 0 else float('inf')
            pdf.add_text(f"Class imbalance (max/min): {imbalance:.2f}")
        else:
            pdf.add_text('Class imbalance: [No valid pixel counts]')
    else:
        pdf.add_text('[Data not available]')

    # 7. Number of ignored or unlabeled pixels
    pdf.add_section_title('7. Number of ignored or unlabeled pixels')
    # Not directly in stats_summary.json, so just mention if not available
    pdf.add_text('See terminal output or extend stats export for ignored/unlabeled pixels.')

    # 8. Visual examples of image and mask overlay
    pdf.add_section_title('8. Visual examples of image and mask overlay')
    pdf.add_text('Below are visual samples: (1) original image, (2) mask, (3) overlay.')
    pdf.add_text('This section presents visual samples for qualitative inspection. For each example, you see: (1) the original RGB image from the Cityscapes dataset, (2) the corresponding segmentation mask with each class represented by a different color, and (3) the overlay, which is the mask blended over the original image. These overlays help visually verify the quality and alignment of the annotations, making it easier to spot labeling errors, class confusion, or annotation inconsistencies.')
    
    overlays_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'overlays')
    overlay_imgs = []
    
    if os.path.exists(overlays_dir):
        for fname in sorted(os.listdir(overlays_dir)):
            if fname.endswith('.png'):
                overlay_imgs.append(os.path.join(overlays_dir, fname))
    
    if overlay_imgs:
        for i, img_path in enumerate(overlay_imgs[:5]):
            pdf.add_image(img_path)
            pdf.add_text(f'Example {i+1}')
    else:
        pdf.add_text('[No overlay images found in outputs/overlays/]')

    # 9. Statistical report files
    pdf.add_section_title('9. Statistical report files')
    pdf.add_text('The following files were generated for further analysis:')
    pdf.add_text('- outputs/statistics/stats_summary.csv (CSV with class statistics)')
    pdf.add_text('- outputs/statistics/stats_summary.json (JSON with class statistics and resolutions)')

    output_pdf = 'outputs/cityscapes_report.pdf'
    pdf.output(output_pdf)
    print(f'PDF report generated at: {output_pdf}')

    pdf.output(output_pdf)
    print(f'PDF report generated at: {output_pdf}')

