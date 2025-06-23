import os
from fpdf import FPDF

def convert_txt_to_pdf(txt_filepath, pdf_filepath):
    """
    Converts a given text file to a PDF file, rendering **text** as bold.

    Args:
        txt_filepath (str): The path to the input .txt file.
        pdf_filepath (str): The path for the output .pdf file.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    line_height = 5  # Define line height for write() and ln()

    # Helper function to process file content and handle bolding
    def process_file_content(file_handle):
        for line in file_handle:
            # Split the line by the bold delimiter '**'
            parts = line.strip().split('**')
            for i, part in enumerate(parts):
                if not part:
                    continue  # Skip empty strings that can result from split

                is_bold = (i % 2 == 1)
                if is_bold:
                    pdf.set_font('Arial', 'B', 10)
                else:
                    pdf.set_font('Arial', '', 10)

                pdf.write(line_height, part)

            pdf.ln(line_height) # Move to the next line in the PDF

    try:
        # Using 'utf-8' encoding is a good default.
        with open(txt_filepath, "r", encoding="utf-8") as f:
            process_file_content(f)
    except UnicodeDecodeError:
        try:
            # Fallback to 'latin-1' if utf-8 fails.
            print(f"UTF-8 decoding failed for {txt_filepath}. Retrying with 'latin-1' encoding.")
            with open(txt_filepath, "r", encoding="latin-1") as f:
                process_file_content(f)
        except Exception as e:
            print(f"Error reading text file {txt_filepath} with fallback encoding: {e}")
            return False
    except Exception as e:
        print(f"Error reading text file {txt_filepath}: {e}")
        return False

    try:
        pdf.output(pdf_filepath)
        print(f"Successfully converted '{txt_filepath}' to '{pdf_filepath}'")
        return True
    except Exception as e:
        print(f"Error writing PDF file {pdf_filepath}: {e}")
        return False

def main():
    """
    Main function to find and convert all .txt files in the 'results' directory.
    """
    results_dir = "results"
    if not os.path.isdir(results_dir):
        print(f"Error: Directory '{results_dir}' not found. Please create it and place your .txt files inside.")
        return

    print(f"Searching for .txt files in '{results_dir}'...")

    for filename in os.listdir(results_dir):
        if filename.endswith(".txt"):
            txt_filepath = os.path.join(results_dir, filename)
            # Create the PDF filename by replacing the extension
            pdf_filename = os.path.splitext(filename)[0] + ".pdf"
            pdf_filepath = os.path.join(results_dir, pdf_filename)

            convert_txt_to_pdf(txt_filepath, pdf_filepath)

if __name__ == "__main__":
    main()
