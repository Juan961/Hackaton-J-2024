import os
import time


from jose import jwt


from exceptions import InvalidTokenException


def validate_access_token(token:str):
    """
    Function to validate token
    :param token:
    :return:
    """

    try:
        secret = os.getenv("Secret")

        payload = jwt.decode(token, secret, algorithms=["HS256"], options={
            "verify_exp": True,
            "verify_sub": True,
            "verify_aud": False,
        })

        return payload
    
    except Exception as e:
        print(e)
        raise InvalidTokenException


def create_access_token(user_info:dict):
    """
    Function to create token
    :param user_id:
    :return:
    """

    secret = os.getenv("Secret")

    payload = {
        "sub": user_info.get("id"),
        "username": user_info.get("username"),
        "_ts": user_info.get("_ts"),
        # 5 minutes
        "exp": int(time.time()) + 300,
    }

    token = jwt.encode(payload, secret, algorithm="HS256")

    return token
