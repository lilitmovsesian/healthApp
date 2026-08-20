from .session import SessionLocal
from .models import UserModel

def seed_user():
	session = SessionLocal()
	try:
		user = UserModel(name="John Doe", email="john.doe@example.com", sex="Male", date_of_birth="1990-01-01")
		session.add(user)
		session.commit()
	finally:
		session.close()


if __name__ == "__main__":
	seed_user()