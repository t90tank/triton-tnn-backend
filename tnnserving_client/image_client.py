import cv2
import numpy as np
import sys
import json
import requests

image = "pics/tiger_cat_2.jpg"

if len(sys.argv) >= 2:
  image = sys.argv[1]

color_img=cv2.imread(image)
data = json.dumps(color_img.tolist())

r = requests.get('http://localhost:8000/v2/models/newtest')
print(r.text)

data = '{"inputs":[{"name":"IN","datatype":"UINT8","shape":[224,224,3],"data":'+data+'}]}'

r = requests.post('http://localhost:8000/v2/models/newtest/infer', data=data)
print(r.text)

data = json.loads(r.text)

answer = data['outputs'][0]['data']

best = 0
for i in range(len(answer)):
  if answer[i] > answer[best]: 
    best = i

label = open("synset.txt", 'r')
result = ""
print(best)
for x in range(best+1):
  result = label.readline()
print(result)