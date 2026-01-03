from paddleocr import PaddleOCR
import sys
import os
import logging

# Suppress Paddle logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

def process_image(img_path):
    if not os.path.exists(img_path):
        print(f"Error: File not found {img_path}")
        return

    # Initialize PaddleOCR (downloads models automatically if needed)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    result = ocr.ocr(img_path)
    
    extracted_text = []
    if result and result[0]:
        for idx in range(len(result)):
            res = result[idx]
            if res:
                for line in res:
                    text = line[1][0]
                    extracted_text.append(text)
    
    # Print joined text for Rust consumption
    print("\n".join(extracted_text))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_ocr.py <image_path>")
        sys.exit(1)
    
    process_image(sys.argv[1])
