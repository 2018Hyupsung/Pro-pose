<div align="center">
  <h3>2022 UHS Computer Engineering Capstone Design </h3>
  <h3>AI기반 자세교정/평가 솔루션 Pro,pose(프로,포즈)</h3>
  <br><br><br>
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
  <br><br><br><br><br>
  
  
  ---
  
  
  <br><br>
  <h3> ★사용 랜드마크★ </h3><br>
  성능향상 및 높은 시인성을 위해 일부 랜드마크의 사용을 제외합니다.
  <br><br>
  <img src="https://mediapipe.dev/images/mobile/pose_tracking_full_body_landmarks.png">
  <br><br>
  0~31번까지의 랜드마크 중 일부를 제외한 11~16, 23~28 총 12개의 랜드마크를 활용합니다.
  <br><br><br><br><br>
  
  
  ---
  
  
  <br><br>
  <h3> ★영상비교 설명★ </h3>
  <br>
  똑같은 영상이 아니고서, 두 가지의 영상을 1:1로 비교한다는 것은 불가능에 가깝습니다.<br>
  첫 번째 이유로는 두 pose 객체가 동일한 좌표평면 상에 놓여있지 않다는 점이고,<br>
  두 번째 이유는 두 영상의 특정 프레임이 같은 자세를 취하고 있지 않을 확률이 높기 때문입니다.<br><br>
  이를 해결하기 위해 3가지의 알고리즘을 활용합니다.<br><br><br><br><br>
  <h3> ★0.Pose Vectorization★ </h3>
  mediapipe는 대부분의 표현을 0~1까지의 정규화된 수치로 나타냅니다.<br>가령, 640x480의 영상이 있고 특정랜드마크의 좌표가 (100,200)이라면<br>
  mediapipe는 이를 (100/640, 200/480) -> (0.18525, 0.41667)로 변환합니다.<br><br>
  <img src="https://user-images.githubusercontent.com/96610952/220829650-ea5e3143-7f2e-40a2-b950-4e3fefa96bcd.png" width="50%"><img src="https://user-images.githubusercontent.com/96610952/220830246-1fb5e215-0926-4445-940a-f90e924f6d76.png" width="50%">
  <br><br><br>
  영상의 넓이와 높이를 곱하여 다시 이를 정규화 전 좌표로 되돌려줍니다.<br>그 후, 두 좌표의 차를 통해 각 랜드마크의 연결부를 벡터화시켜줍니다.<br><br>
  <img src="https://user-images.githubusercontent.com/96610952/220831634-2f02bed1-3941-44e1-b607-b58c2b802c31.png" width="50%">
  <br><br><br><br><br>
  <h3> ★1.L2 Norm/Normalization★ </h3>
  <br><br>
  두 영상이 같은 자세를 취하고 있더라도, 프레임 속 인물의 위치나 키, 팔다리 길이의 차이 등의<br>이유로 인해 좌표는 언제나 달라질 수 있습니다.<br><br>
  두 개의 서로 다른 벡터를 같은 환경에서 비교하기 위해서 L2 Norm을 통해 벡터의 크기를 계산하고,<br>정규화(Normalization)을 통해 벡터의 크기를 0~1로 통일합니다.<br><br>
  <img src="https://user-images.githubusercontent.com/96610952/220835072-1563d4a9-0065-446b-83d0-b57e34cb7810.png"><br>
  <특정 2차원 벡터 u = (x,y)의 L2 Normalization>
  <br><br><br><br><br>
  <h3> ★2.Cosine Similarity★ </h3>
  <br><br>
  L2 정규화를 통해 벡터를 같은 조건에서 비교할 수 있게 되었으나,<br>벡터의 크기가 다를 경우(두 영상에서 사람 객체의 원근감이나 팔다리 길이 차이로 인한 경우) 컴퓨터는 이 두 벡터를<br>
  유사하지 않다고 판단할 것입니다.<br><br>
  <img src="http://matrix.skku.ac.kr/math4AI-tools/cosine_similarity/PICA5CF.png"><br><br>
  코사인 유사도(Cosine Similarity)를 활용하면 두 벡터의 크기와 거리는 무시하고 오로지 두 벡터의 방향을 통해 유사도를 판단하게 됩니다.<br><br>
  <img src="https://wikidocs.net/images/page/24603/코사인유사도.PNG"><br><br>
  <img src="https://user-images.githubusercontent.com/96610952/220837470-933954a6-fc12-469b-8d66-4509463655d6.png"><br>
  < i 번째 부위에 대한 두 벡터의 코사인 유사도>
  <br><br><br><br><br>
  <h3> ★3.Euclidean Distance★ </h3>
  <br><br>
  두 벡터 사이의 거리를 구하는 유클리디안 거리 공식을 통해, 앞서 구한 코사인유사도의 값을 정량화하고,<br>이를 기준으로 각 부위별 스코어를 계산합니다.<br><br>
  <img src="https://user-images.githubusercontent.com/96610952/220840794-141d4c86-332f-4f1e-9902-faaa75809c6b.png">
  <br><br><br><br><br>
  <h3> ★4.Dynamic Time Wraping★ </h3>
  <br>
  두 영상을 첫 프레임부터 순차적으로 비교할 때, 특정동작의 흐름이 모든 초에서 동일할 순 없습니다.<br><br>
  가령 윗몸일으키키를 할 때, 누군가는 1개의 동작을 완료하는데에 1초가 걸릴 수 있으나,<br>혹자는 1개 동작을 완료하는데에 2초의 시간이 소요될 수 있습니다.<br>
  이 때 DTW(Dynamic Time wraping)을 이용해<br><br>
  1) A영상의 1초 때의 프레임<br>
  2) B영상의 0~2초 사이의 모든 프레임<br>
  의 유클리디안 거리를 모두 비교하여 DTW의 거리가 가장 짧은(가장 유사도가 비슷한) 두 프레임을 비교/분석합니다.<br><br>
  <img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*5TRtfoKOyOgIu4QkoB8bFg.png"><br>
</div>



