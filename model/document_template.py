from db.database import Base
from sqlalchemy.orm import mapped_column
from sqlalchemy import String, Integer, Date
from sqlalchemy.dialects.postgresql import JSONB


class DocumentTemplate(Base):
    __tablename__ = "document_template"
    id = mapped_column(Integer, primary_key=True)
    document_type = mapped_column(String(120), nullable=False)
    country = mapped_column(String(120), nullable=True)
    state = mapped_column(String(120), nullable=True)
    edition = mapped_column(Date, nullable=True)
    document_name = mapped_column(String(120), nullable=False)
    img_path = mapped_column(String(120))
    fields = mapped_column(JSONB)
    #anchors = mapped_column(JSONB, default=list)
