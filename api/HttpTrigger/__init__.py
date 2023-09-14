import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler
import os
import ssl
from dotenv import load_dotenv
import urllib.request
import azure.functions as func
import json

import openai
from openai import util

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
azureml_api_key = os.getenv("AZUREML_API_KEY")

logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(connection_string=os.environ['APPINSIGHT_CONNECT_KEY']))

def allowSelfSignedHttps(allowed):#SSL証明書の検証をバイパス
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        messages = req_body.get('messages')

    if not messages:
        return func.HttpResponse("Please pass a message in the request body", status_code=400)

    # Get the latest user message
    question = messages[-1]["content"] if messages[-1]["role"] == "user" else None

    if not question:
        return func.HttpResponse("No user message found in the request body", status_code=400)

    response = call_openai_api(question, req)
    return func.HttpResponse(str(response))

def call_openai_api(question:str, req:func.HttpRequest):
    logging.info(f"Got Question: {question}")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "You are best assistant ever!"},
            {"role": "user", "content": question}],
        functions=[
            {
                "name":"answer_question_about_teamspirits",
                "description":"Prompt flowのエンドポイントから、Teamspiritsに関する社内マニュアルを参照して回答する。",
                "parameters":{
                    "type":"object",
                    "properties":{
                        'message':{"type":"string"},
                    },
                    "required":["message"],
                }
            }
        ],
        function_call="auto",
    )
    message=response["choices"][0]["message"]

    # STEP2: モデルが関数を呼び出したいかどうかを確認
    if message.get("function_call"):
        function_name = message["function_call"]["name"]
        arguments = json.loads(message["function_call"]["arguments"])
    
        if function_name == "answer_question_about_teamspirits":
            function_response = answer_question_about_teamspirits(
                message=arguments.get("message"),
                azureml_api_key=os.getenv("AZUREML_API_KEY")
            )
        else:
            raise NotImplementedError()
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
             # get user input
             
            messages=[
                {"role": "user", "content": question},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": str(function_response),
                },
            ],
        )
        return second_response
    else:
        return response

def answer_question_about_teamspirits(message:str, azureml_api_key, allow_self_signed=True):#関数内容(HTTPリクエスト内構造の定義)
    allowSelfSignedHttps(allow_self_signed)

    data={'question':message}

    body = str.encode(json.dumps(data))
    url = 'https://mydeployment.eastus2.inference.ml.azure.com/score'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {azureml_api_key}',
        'azureml-model-deployment': 'mydeployment-1'
    }
    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        if isinstance(result, bytes):
            result = result.decode('utf-8')
            logging.info(f"Type of 'result':{type(result)}")
        return json.dumps(result, ensure_ascii=False)
    except urllib.error.HTTPError as error:
        print(f"The request failed with status code: {error.code}")
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
        return None
