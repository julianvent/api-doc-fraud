import tempfile

from fastapi import UploadFile
import shutil
from pathlib import Path

from service.preprocessor import preprocessor
from service.tampering import tampering
from service.metadata import metadata
from service.ocr import ocr

DEST_PATH = "files"


def process_file(file_path: str):
    # Call the process file function in pipeline
    pass


def process_files(files: list[UploadFile], id: str):
    file_paths = []
    for file in files:
        file_paths.append(upload_file(file, id))

    # Files already on local dir imgs/
    # TODO: Call the service for preprocessing images
    preprocessor.process_batch(file_paths=file_paths)

    """
    Maybe you can save the file paths for the processed images
    processed_file_paths = preprocessor.process_batch(file_paths=file_paths)
    """

    # Next, perform metadata, tampering, and OCR analysis
    metadata.extract(file_paths=file_paths)
    tampering.analyze()

    ocr.process(
        file_path=file_paths
    )  # Check whether to process a batch or one single image


def upload_file(file: UploadFile, id: str) -> str:
    tmp_dir  = Path(tempfile.gettempdir()) / "ocr_tmp" / id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    file_path = tmp_dir / file.filename
    with file_path.open(mode="wb") as buffer:
        try:
            shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            print(f" x Error writing file: {file.filename} - {e}")
            return ""

    return file_path
