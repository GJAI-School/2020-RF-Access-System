# RF_access_system (2020.11.13)

  - 📫  프로젝트 : RF 전자 명부 시스템 개발
  - 📫  프로젝트 개요 : 라젠의 RF기술과 인공지능 기반의 얼굴 및 정맥 인증을 통한 전자 명부 시스템 개발 

## 🎈 Makers(어바웃타임 팀)
- 고정환 : `코드분석`, `하드웨어`
- 강민지 : `코드분석`, `성능확인`
- 박성찬 : `코드분석`, `데이터전처리`
- 이윤환 : `모델링`, `데이터전처리`

## 💡 프로젝트 배경 
- 기존 개인정보 관리의 허점 : 기록된 정보를 모든 사람들이 볼 수 있어서 개인정보가 쉽게 노출
- 수집한 정보 관리의 어려움 : 출입 기록에 대한 정보를 종이 들에 작성하여 관리의 어려움
- 비대면이 아닌 대면 기록의 한계 : QR코드 인식 후, 사람을 통해 체온을 측정하는 등 대면 접촉 발생



## 💡 프로젝트 프로세스
![image](https://user-images.githubusercontent.com/58651942/100871544-ce66db80-34e3-11eb-91f3-db4dc02a8f27.png)



## 💡 프로젝트 과정
- 프로젝트는 두 팀이 함께 협업했음
- 우리팀(어바웃타임)은 얼굴 인식과 전자출입명부 부분을 맡아서 진행함

![image](https://user-images.githubusercontent.com/58651942/100871613-e50d3280-34e3-11eb-85b1-a912fa4d6216.png)

## 💡 프로젝트 일정
![image](https://user-images.githubusercontent.com/58651942/100842919-ea0abb80-34bc-11eb-88e7-7cb78b44db6b.png)
 

## 💡 Face-Detection
![image](https://user-images.githubusercontent.com/58651942/100843233-68fff400-34bd-11eb-8093-a9243f0f0e67.png)

## 💡 Face-Recognization
![image](https://user-images.githubusercontent.com/58651942/100843647-03603780-34be-11eb-95eb-f2690c5caf5a.png)

## 💡 Face-Recognization Modeling

![image](https://user-images.githubusercontent.com/58651942/100871683-040bc480-34e4-11eb-8a14-d7322926b958.png)


## 💡 성능확인(ROC, check_thresh_hold)
- RF access system의 얼굴 인식 모델로 `facenet_abouttime`을 사용</br>

![image](https://user-images.githubusercontent.com/58651942/100871762-1f76cf80-34e4-11eb-9686-1d2c48cf5c21.png)

## 💡 전자출입명부
![image](https://user-images.githubusercontent.com/58651942/100846221-98b0fb00-34c1-11eb-8ed4-e4d413df847a.png)



## 💡 결과분석 - 피드백 및 보완점
- 전이 학습에 사용된 데이터 개수는 기존 facenet을 모델링할 때 사용한 데이터 개수의 0.01% 수준임. 더 많은 데이터로 더 많은 시간동안 학습시키면 모델 성능 개선 폭이 더 커질 것으로 예상됨. 
- 마스크를 착용한 데이터의 학습을 통해, 마스크를 벗지 않아도 안면 인식이 가능한 서비스로 확장할 수 있음.
