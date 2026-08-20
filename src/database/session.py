from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(
	"sqlite:///database.db"
)

SessionLocal = sessionmaker(bind=engine)