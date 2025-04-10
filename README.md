# Camera_Pose_Estimation_and_AR
Camera Pose Estimation and AR

---

## 코드 설명

### 1. 입력 및 캘리브레이션 데이터 설정

- **영상 파일 및 카메라 캘리브레이션 정보**
  - `video_file`에는 입력 영상 파일(예: `'myvideo.avi'`) 경로를 지정함.
  - `K`는 미리 캘리브레이션한 카메라 매트릭스로,  
    - fx = 588.095, fy = 593.846  
    - cx = 633.172, cy = 353.322 값을 포함함.
  - `dist_coeff`는 왜곡 계수 배열임.
  
- **체스보드 패턴 및 실제 크기**
  - `board_pattern`은 체스보드 코너 개수를 `(10, 7)`로 설정함.
  - `board_cellsize`는 각 체스보드 셀의 실제 크기를 0.025미터로 지정함.
  - `board_criteria`는 체스보드 코너 검출 시 사용되는 추가 옵션(적응형 이진화, 정규화 등)을 지정함.

### 2. 체스보드 3D 점 생성

- 체스보드 상의 모든 코너(3D 점)는 Z=0 평면에 위치하며,  
  각 점은 `[c, r, 0]` 형식의 좌표로, 실제 크기(board_cellsize)가 곱해진 값으로 계산됨.

```python
obj_points = board_cellsize * np.array(
    [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])],
    dtype=np.float32)
```

### 3. AR 피라미드 모형 정의

- **모형의 3D 점 정의**
  - 피라미드 모형은 바닥에 정사각형 형태의 4개 모서리와, 중앙 위쪽에 있는 apex(정점)를 포함함.
  - `L`은 피라미드의 바닥 변 길이(및 높이 스케일)로, 보드 셀 크기의 3배로 설정됨.
  
```python
L = board_cellsize * 3
pyramid_pts = np.array([
    [0, 0, 0],         # 바닥 모서리 1
    [L, 0, 0],         # 바닥 모서리 2
    [L, L, 0],         # 바닥 모서리 3
    [0, L, 0],         # 바닥 모서리 4
    [L/2, L/2, -L]     # 피라미드 꼭짓점 (바닥에서 위로 -L, 즉 위쪽으로)
], dtype=np.float32)
```

### 4. 영상 처리 및 포즈 추정

- **영상 읽기**
  - OpenCV의 `VideoCapture`로 지정한 영상을 열고, 각 프레임을 처리함.
  
- **체스보드 코너 검출**
  - 각 영상 프레임마다 `cv.findChessboardCorners()`를 이용하여 체스보드 코너(2D 이미지 좌표)를 검출함.
  
- **카메라 포즈 추정**
  - 검출된 2D 코너와 미리 준비한 3D 체스보드 점을 사용하여, `cv.solvePnP()` 함수를 통해 회전 벡터(`rvec`)와 평행 이동 벡터(`tvec`)를 추정함.

```python
success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
if success:
    ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)
```

### 5. AR 물체(피라미드) 투영 및 그리기

- **3D 점 투영**
  - `cv.projectPoints()`를 사용하여 피라미드 모형의 3D 점을 영상상의 2D 좌표로 투영함.
  
- **모형 그리기**
  - 바닥면(정사각형)은 `cv.polylines()`로 닫힌 다각형으로 그리며,  
  - 각 바닥 모서리에서 피라미드 정점까지 `cv.line()`을 이용해 선을 그려 피라미드 모형을 완성함.

```python
projected_pts, _ = cv.projectPoints(pyramid_pts, rvec, tvec, K, dist_coeff)
projected_pts = np.int32(projected_pts).reshape(-1, 2)
cv.polylines(img, [projected_pts[:4]], isClosed=True, color=(255, 0, 0), thickness=2)
for i in range(4):
    cv.line(img, tuple(projected_pts[i]), tuple(projected_pts[4]), color=(0, 255, 0), thickness=2)
```

### 6. 카메라 위치 출력

- **카메라 위치 계산**
  - `cv.Rodrigues()`를 이용해 회전 행렬(R)을 얻은 후, 카메라 중심 위치를 `-R.T @ tvec`로 계산함.
  - 계산된 카메라 위치는 영상에 텍스트로 표시됨.

```python
R, _ = cv.Rodrigues(rvec)
cam_pos = (-R.T @ tvec).flatten()
info = f'XYZ: [{cam_pos[0]:.3f} {cam_pos[1]:.3f} {cam_pos[2]:.3f}]'
cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
```

### 7. 결과 영상 표시 및 종료 처리

- **결과 영상 창에 출력**
  - 최종적으로 AR 모형과 카메라 위치 정보가 포함된 결과 영상을 OpenCV 창에 출력함.
  
- **키 입력 처리**
  - ESC 키를 누르면 프로그램이 종료됨.

```python
cv.imshow('Pose Estimation (Chessboard)', img)
key = cv.waitKey(10)
if key == ord(' '):
    key = cv.waitKey()
if key == 27:  # ESC 키로 종료
    break
```

---

### 결과사진
<img src="https://github.com/user-attachments/assets/c4656b2b-f242-47b9-ab17-1c520c47a8c7" height=400>
