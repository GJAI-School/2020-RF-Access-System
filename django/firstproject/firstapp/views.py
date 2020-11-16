from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import json
import cv2


# Create your views here.
def home(request):
    data = np.fromstring(request.body, dtype = 'uint8')
    #data를 디코딩한다.
    try:
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        cv2.imwrite('cam.jpg', frame)
        
    except:
        pass
    return render(request, 'home.html')
