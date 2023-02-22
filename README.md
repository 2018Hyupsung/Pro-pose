<div align="center">
  <h3>2022 UHS Computer Engineering Capstone Design </h3>
  <h3>AI기반 자세교정/평가 솔루션 Pro,pose(프로,포즈)</h3>
  <지도교수><br>홍석주<br><br><제작인원><br><김봉준/장준영/서재원>
</div>
<br><br><br><br><br>



<div align="center">
  <img src="https://user-images.githubusercontent.com/96610952/220712126-080e101f-2260-4445-b3d8-35dd28ae65c7.svg" width="600px">
</div>
  
<div align="center">
  <br><br><br>
  <h3> ★LOGO★ </h3>
  <br>
  <br>
  <img src="https://user-images.githubusercontent.com/96610952/220722785-e132be69-907b-4323-9584-3c2e999600fd.png" width="170px"/>
  <br>
  <br>
  "미숙하지만 뜨거운 열정을 가진 멘티와 식은 듯 보이지만 능숙한 능력을 지닌 멘토 사이를 이어주는 퍼즐"<br><br>
AI는 한 없이 냉정하고 차가워보이지만, 언제나 그 자리에서 끊임없이 멘티에게 평가와 조언을 아까지않는 멘토처럼 보이기도 합니다.<br>
열정과 이성을 불과 물의 색상으로 표현하였고, 워드마크의 위치와 점, 선들은 Human pose estimation 분야에서 빠질 수 없는 랜드마크 표현을 참고하였습니다.<br><br><br>
</div>


---
<br>
<div align="center">
  <div align="center">
    <img src="https://img.shields.io/badge/python-3.9.2rc1-blue">
    <br>
    <img src="https://img.shields.io/badge/mediapipe-0.9.0.1-pink">
    <img src="https://img.shields.io/badge/opencv-4.7.0.68-pink">
    <img src="https://img.shields.io/badge/ffmpeg-0.2.0-pink">
    <img src="https://img.shields.io/badge/numpy-1.24.1-pink">
    <img src="https://img.shields.io/badge/pandas-1.5.3-pink">
    <br>
    <img src="https://img.shields.io/badge/dtaidistance-2.3.10-pink">
    <img src="https://img.shields.io/badge/-with edited dtw.py-pink">
    <br>
    <img src="https://img.shields.io/badge/node.js-18.14.2-green">
    <img src="https://img.shields.io/badge/react-18.2.0-green">
  </div>
 </div>
<br>

---
  
<br><br>
<div align="center">
  <h3> ★Human Pose Estimation★ </h3>
  Human Pose Estimation은 컴퓨터비전의 중요 과제 중 하나로써<br>사람의 관절마다 key point를 구성하여 연결한 뒤, 사람 객체를 찾아내어 추적하는 것을 말합니다.
  <br>
  <br>
  현재 다양한 종류의 Estimation 모델들이 출시되어 있습니다.<br>본 프로젝트에서는 각 영상마다 단일 객체를 인식한다는 점, 높은 GPU처리성능 없이 CPU로 분석이 가능하다는 점, 높은 프레임의 결과를 보여
  준다는 점을 들어<br>mediapipe 모델을 활용합니다.<br>
  <img src="https://mediapipe.dev/assets/img/brand.svg">
  <br><br><br>
  <h3> ★사용 랜드마크★ </h3>
  성능향상 및 높은 시인성을 위해 일부 랜드마크의 사용을 제외합니다.
  <br><br>
  <img src="https://mediapipe.dev/images/mobile/pose_tracking_full_body_landmarks.png">
  <br><br>
  0~31번까지의 랜드마크 중 일부를 제외한 11~16, 23~28 총 12개의 랜드마크를 활용합니다.
  <br><br><br><br>
  <h3> ★영상비교 설명★ </h3>
  <br>
  똑같은 영상이 아니고서, 두 가지의 영상을 1:1로 비교한다는 것은 불가능에 가깝습니다.<br>
  첫 번째 이유로는 두 pose 객체가 동일한 좌표평면 상에 놓여있지 않다는 점이고,<br>
  두 번째 이유는 두 영상의 특정 프레임이 같은 자세를 취하고 있지 않을 확률이 높기 때문입니다.<br><br>
  이를 해결하기 위해 3가지의 알고리즘을 활용합니다.<br><br><br>
  <h3> ★0.Pose Vectorization★ </h3>
  영상에서의 
  <h3> ★1.L2 Norm/Regularization★ </h3>
  

</div>
