# test_ocr.py
import sys
sys.path.append(".")

from service.ocr.ocr import process

result = process(
    file_path="files/docu/processed/visaAndres_p1.png",
    document_type="visa"
)

import json
print(json.dumps(result, ensure_ascii=False, indent=2))