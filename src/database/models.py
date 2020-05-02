from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class File(Base):
    __tablename__ = 'file'

    id = Column(Integer, primary_key=True)
    absolute_path = Column(String)
    size = Column(Integer)
    last_modified = Column(TIMESTAMP)
    created = Column(TIMESTAMP)


class Features(Base):
    __tablename__ = 'features'

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('file.id'))
    width = Column(Integer)
    height = Column(Integer)
    feature_vector = Column(String)
    blurriness = Column(Float)


class Object(Base):
    __tablename__ = 'object'

    id = Column(Integer, primary_key=True)
    name = Column(String)


class Detection(Base):
    __tablename__ = 'detection'

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('file.id'))
    object_id = Column(Integer, ForeignKey('object.id'))
    bbox = Column(String)


def create_tables(engine):
    Base.metadata.create_all(engine)