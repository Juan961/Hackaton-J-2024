import json
import unittest


from azure.functions import HttpRequest


from functions.predict import predict
from functions.login import login  # Assuming you have a login function

class TestPrediction(unittest.TestCase):
    def setUp(self):
        login_req = HttpRequest(
            method='POST',
            body=json.dumps({
                'username': 'admin',
                'password': 'password-secured'
            }).encode('utf-8'),
            url='/api/login',
            headers={
                'Content-Type': 'application/json'
            }
        )
        login_func = login.build().get_user_function()
        login_resp = login_func(login_req)
        self.token = json.loads(login_resp.get_body().decode())['accessToken']

    def test_invalid_token(self):
        req = HttpRequest(
            method='POST',
            body=json.dumps({}).encode('utf-8'),
            url='/api/predict/classification',
            headers={
                'Content-Type': 'application/json',
                'Authorization': "invalid_token"
            },
            route_params={
                "model": "classification"
            }
        )

        func_call = predict.build().get_user_function()

        resp = func_call(req)

        self.assertEqual(resp.status_code, 401)

    def test_invalid_model(self):
        req = HttpRequest(
            method='POST',
            body=json.dumps({}).encode('utf-8'), 
            url='/api/predict/invalid',
            headers={
                'Content-Type': 'application/json',
                'Authorization': self.token
            },
            route_params={
                "model": "invalid"
            }
        )

        func_call = predict.build().get_user_function()

        resp = func_call(req)

        self.assertEqual(resp.status_code, 400)

    def test_predict_classification_invalid_data(self):
        req = HttpRequest(
            method='POST',
            body=json.dumps({
                "Sunlight_Hours": 4.033,
                "Temperature": 28.91,
                "Humidity": 52.42,
                "Soil_Type": "sandy",
                "Water_Frequency": "weekly",
                "asd": "asd",
                "as": "asdqwe",
            }).encode('utf-8'),
            url='/api/predict/classification',
            headers={
                'Content-Type': 'application/json',
                'Authorization': self.token
            },
            route_params={
                "model": "classification"
            }
        )

        func_call = predict.build().get_user_function()

        resp = func_call(req)

        self.assertEqual(resp.status_code, 400)

    def test_predict_classification(self):
        req = HttpRequest(
            method='POST',
            body=json.dumps({
                "Sunlight_Hours": 4.033,
                "Temperature": 28.91,
                "Humidity": 52.42,
                "Soil_Type": "sandy",
                "Water_Frequency": "weekly",
                "Fertilizer_Type": "organic",
            }).encode('utf-8'),
            url='/api/predict/classification',
            headers={
                'Content-Type': 'application/json',
                'Authorization': self.token
            },
            route_params={
                "model": "classification"
            }
        )

        func_call = predict.build().get_user_function()

        resp = func_call(req)

        self.assertEqual(resp.status_code, 200)

        # Must come the response with "response" and "growing" keys
        self.assertIn("response", json.loads(resp.get_body().decode()))
        self.assertIn("growing", json.loads(resp.get_body().decode()))

        # response must be a string growing must be a boolean
        self.assertIsInstance(json.loads(resp.get_body().decode())["response"], str)
        self.assertIsInstance(json.loads(resp.get_body().decode())["growing"], bool)

    def test_predict_image(self):
        req = HttpRequest(
            method='POST',
            body=json.dumps({}).encode('utf-8'), 
            url='/api/predict/image',
            headers={
                'Content-Type': 'application/json',
                'Authorization': self.token
            },
            route_params={
                "model": "image"
            }
        )

        func_call = predict.build().get_user_function()

        resp = func_call(req)

        self.assertEqual(resp.status_code, 200)

        # Must come the response with "response" and "plant" keys
        self.assertIn("response", json.loads(resp.get_body().decode()))
        self.assertIn("plant", json.loads(resp.get_body().decode()))

        # response must be a string plant must be a str
        self.assertIsInstance(json.loads(resp.get_body().decode())["response"], str)
        self.assertIsInstance(json.loads(resp.get_body().decode())["plant"], str)
