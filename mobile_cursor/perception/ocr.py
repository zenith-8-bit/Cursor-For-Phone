class OCRReader:
    def read(self, image_path):
        try:
            import pytesseract
            from PIL import Image
            text = pytesseract.image_to_string(Image.open(image_path))
            return [x.strip() for x in text.splitlines() if x.strip()]
        except Exception:
            return []
