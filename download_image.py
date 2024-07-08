import requests

url = "http://127.0.0.1:8000/find_similar/"
file_path = "https://i.pinimg.com/736x/21/cc/30/21cc30938777c974b49752464975cfa3.jpg"

with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

print(response.json())
