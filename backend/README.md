# Hackaton - Backend

The backend manages login requests using JWT tokens. The tokens are used and validated in the endpoint of prediction, that has the job of processing the data using the ML model and returning the predictions as a JSON response.

## 🐍 Suggested Pyhton version
```
3.11.*
```

## 📐 Setup

Linux, MacOs
``` bash
$ python3 -m venv venv
$ source venv/bin/activate
```

Windows
``` bash
$ py -m venv venv
$ source venv/scripts/activate
```

## 🚀 Dev

``` bash
$ func start
```

## 🧪 Test

``` bash
$ pytest tests
```
