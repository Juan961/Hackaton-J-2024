import os


from cryptography.fernet import Fernet


from exceptions import InvalidUserException


class User:
    @staticmethod
    def login(username: str, password: str):
        admin_username = os.getenv("AdminUsername")
        admin_password = os.getenv("AdminPassword")

        secret = os.getenv("Secret")

        fernet = Fernet(secret)

        admin_username_decrypted = fernet.decrypt(admin_username.encode()).decode()
        admin_password_decrypted = fernet.decrypt(admin_password.encode()).decode()

        if username != admin_username_decrypted or password != admin_password_decrypted:
            raise InvalidUserException

        return {
            "id": "asdd-1234-1234-1234",
            "username": "admin",
            "_ts": 1
        }
