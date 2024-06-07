from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
import openai

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
cliente  = OpenAI(api_key=api_key)

def is_openai_api_status(api_key):
    openai.api_key = api_key
    url = "https://api.openai.com/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  
        return True, response  
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")
        
    return False,response 

status, resposta_status = is_openai_api_status(api_key)

print(f'{status}'+'\n')
print(f'{resposta_status}'+'\n')

def bot_teste_status(api_key, prompt):
    if not is_openai_api_status(api_key):
        return "API fora do ar"
    
    try:
        resposta = cliente.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um bot que verifica o status da API da openai"},
                {"role": "user", "content": prompt}
            ]
        )
        print(resposta.choices[0].message.content)
    except openai.error.OpenAIError as e:
        print(f"OpenAI error: {e}")
        return str(e)
    except openai.APIConnectionError as e:
        print(f"Failed to connect to OpenAI API: {e}")
        return str(e)
    except openai.RateLimitError as e:
        print(f"OpenAI API request exceeded rate limit: {e}")
        return str(e)
    

prompt = "Me informe apenas se a API da apenai está com status ativo"

bot_teste_status(api_key, prompt) #Tratar os códigos de status de erro que existem na documentação https://platform.openai.com/docs/guides/error-codes/api-errors