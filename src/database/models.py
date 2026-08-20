from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, ForeignKey, Integer, String

class BaseModel(DeclarativeBase):
    pass

class UserModel(BaseModel):
    __tablename__ = "users"

    id: Column[int] = Column(Integer, primary_key=True)
    name: Column[str] = Column(String(100))
    email: Column[str] = Column(String(100))
    date_of_birth: Column[str] = Column(String(10))
    sex: Column[str] = Column(String(10))

class DocumentModel(BaseModel):
    __tablename__ = "documents"

    id: Column[int] = Column(Integer, primary_key=True)
    user_id: Column[int] = Column(Integer, ForeignKey("users.id"))
    content: Column[str] = Column(String)

class BloodTestResultModel(BaseModel):
    __tablename__ = "blood_test_results"

    id: Column[int] = Column(Integer, primary_key=True)
    document_id: Column[int] = Column(Integer, ForeignKey("documents.id"))
    user_id: Column[int] = Column(Integer, ForeignKey("users.id"))
    test_date: Column[str] = Column(String(50))
    test_name: Column[str] = Column(String(100))
    value: Column[str] = Column(String(50))
    unit: Column[str] = Column(String(20))