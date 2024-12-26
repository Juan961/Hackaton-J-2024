from azure.functions import FunctionApp


from functions.login import login_blueprint
from functions.predict import predict_blueprint


app = FunctionApp()


app.register_functions(login_blueprint)
app.register_functions(predict_blueprint)
