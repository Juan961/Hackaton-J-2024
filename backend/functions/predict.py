import json
import logging
import traceback


from http import HTTPStatus
from azure.functions import Blueprint, AuthLevel, HttpRequest, HttpResponse, HttpMethod


from machine.image import predict_image
from machine.data_growth import predict_classification

from auth import validate_access_token
from exceptions import InvalidTokenException


predict_blueprint = Blueprint()


@predict_blueprint.function_name("Predict")
@predict_blueprint.route(route="predict/{model}", auth_level=AuthLevel.ANONYMOUS, methods=(HttpMethod.POST,))
def predict(req: HttpRequest) -> HttpResponse:
    try:
        logging.info('Predict: Started')

        model_to_predict = req.route_params.get("model")

        if model_to_predict not in ["classification", "image"]:
            return HttpResponse(
                body=json.dumps({
                    "error": "Model not found"
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                status_code=HTTPStatus.BAD_REQUEST
            )

        token = req.headers.get("Authorization", "")

        validate_access_token(token)

        body = req.get_json()

        # Run the function predict_{model} with eval
        response = eval(f"predict_{model_to_predict}")(body)

        return HttpResponse(
            body=json.dumps(response).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            status_code=HTTPStatus.OK
        )

    except InvalidTokenException as e:
        return HttpResponse(
            body=json.dumps({
                "error": "Invalid token"
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            status_code=HTTPStatus.UNAUTHORIZED
        )

    except ValueError as e:
        return HttpResponse(
            body=json.dumps({
                "error": e.args[0]
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
