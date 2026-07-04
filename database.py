from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


# Database URL configuration
database_url = os.environ.get('DATABASE_URL') or 'mysql+pymysql://root:123abc@localhost/careerly'

# Create SQLAlchemy engine
engine = create_engine(database_url, echo=True, pool_pre_ping=True, pool_recycle=300, pool_timeout=20, max_overflow=0)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

