from fastapi import UploadFile
import shutil
from pathlib import Path

from service.preprocessor import preprocessor

DEST_PATH = "files"


def process_file(file_path: str):
    # Call the process file function in pipeline
    pass


def process_files(files: list[UploadFile], id: str):
    for file in files:
        upload_file(file, id)

    # Files already on local dir imgs/
    # TODO: Call the service for preprocessing images
    preprocessor.process_batch()


def upload_file(file: UploadFile, id: str) -> str:
    dest_path = Path(f"{DEST_PATH}/{id}")
    dest_path.mkdir(parents=True, exist_ok=True)

    file_path = dest_path / file.filename
    with file_path.open(mode="wb") as buffer:
        try:
            shutil.copyfileobj(file.file, buffer)
            print(f" > File saved: {file_path}")
        except Exception as e:
            print(f" x Error writing file: {file.filename} - {e}")

    return file_path
