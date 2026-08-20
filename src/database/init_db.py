from .session import engine
from .models import BaseModel

def create_tables():
    BaseModel.metadata.create_all(engine)

if __name__ == "__main__":
	create_tables()