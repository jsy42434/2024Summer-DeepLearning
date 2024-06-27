import requests
import json

# json 구성
# model : 사용하고자하는 라마 모델 이름
# prompt : 답변을 받길 원하는 질문
llama_requset = {
    'model': 'llama3',
    'prompt': 'what is json'
}

response = requests.post('http://localhost:11434/api/generate', json=llama_requset)
#print(response.text) #순수 json답변을 보고 싶을때만 주석 해제

# 서버로부터의 JSON 응답을 문자열로 가져와서 분할
responses = response.text.strip().split('\n')

# response 키의 값들을 추출하여 문장 생성
sentence = ''.join([json.loads(response)['response'] for response in responses if 'response' in json.loads(response)])

# 문장 출력
print(sentence)
