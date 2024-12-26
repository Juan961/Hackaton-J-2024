import json
import logging
import traceback


from http import HTTPStatus
from azure.functions import Blueprint, AuthLevel, HttpRequest, HttpResponse, HttpMethod


from models.user import User
from auth import create_access_token
from exceptions import InvalidUserException


login_blueprint = Blueprint()


@login_blueprint.function_name("Login")
@login_blueprint.route(route="login", auth_level=AuthLevel.ANONYMOUS, methods=(HttpMethod.POST,))
def login(req: HttpRequest) -> HttpResponse:
    try:
        logging.info('Login: Started')

        body = req.get_json()

        username = body.get("username")
        password = body.get("password")

        db_user = User.login(username, password)

        token = create_access_token(db_user)

        return HttpResponse(
            body=json.dumps({
                "accessToken": token
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            status_code=HTTPStatus.OK
        )

    except InvalidUserException as e:
        return HttpResponse(
            body=json.dumps({
                "error": "User not found"
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            status_code=HTTPStatus.BAD_REQUEST  
        )

    except Exception as e:
        exception_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logging.error(exception_str)
        return HttpResponse(
            body=json.dumps({
                "error": "Server error"
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
