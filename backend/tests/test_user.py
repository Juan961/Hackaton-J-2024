import os
import json
import unittest


from azure.functions import HttpRequest


from functions.login import login


class TestUser(unittest.TestCase):
    def test_login_valid_user(self):
        req = HttpRequest(
            method='POST',
            body=json.dumps({
                "username": "admin",
                "password": "password-secured"
            }).encode('utf-8'), 
            url='/api/login',
            headers={
                'Content-Type': 'application/json',
            }
        )

        func_call = login.build().get_user_function()

        resp = func_call(req)

        self.assertEqual(resp.status_code, 200)

    def test_login_user_invalid(self):
        req = HttpRequest(
            method='POST',
            body=json.dumps({
                "username": "admin",
                "password": "password-insecured"
            }).encode('utf-8'), 
            url='/api/login',
            headers={
                'Content-Type': 'application/json',
            }
        )

        func_call = login.build().get_user_function()

        resp = func_call(req)

        self.assertEqual(resp.status_code, 400)
