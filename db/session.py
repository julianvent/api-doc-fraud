from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("postgresql://postgres:root@localhost:5432/antifraud")

Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
