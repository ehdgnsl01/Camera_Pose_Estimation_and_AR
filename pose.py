import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = 'myvideo.avi'
K = np.array([[588.09499804, 0, 633.1719751],
              [0, 593.84617224, 353.32181687],
              [0, 0, 1]])  # Derived from camera_calibration.py
dist_coeff = np.array([0.10588554, -0.19755432, -0.00568951, -0.00416748, 0.12764793])
board_pattern = (10, 7)
board_cellsize = 0.025  # 단위: 미터
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Prepare a 3D pyramid for AR overlay
# L는 피라미드의 바닥 변 길이 (및 높이 스케일)로 설정
L = board_cellsize * 3
pyramid_pts = np.array([
    [0, 0, 0],         # 바닥 모서리 1
    [L, 0, 0],         # 바닥 모서리 2
    [L, L, 0],         # 바닥 모서리 3
    [0, L, 0],         # 바닥 모서리 4
    [L/2, L/2, -L]     # 피라미드 꼭짓점 (바닥에서 위로 -L, 즉 위쪽으로)
], dtype=np.float32)

# Prepare 3D points on a chessboard for pose estimation
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation and AR overlay
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # 체스보드 코너 검출 (포즈 추정용)
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        # 포즈 추정: 3D 체스보드 점과 2D 검출 코너를 이용
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # 3D 피라미드 모델의 점들을 영상상의 2D 점으로 투영
        projected_pts, _ = cv.projectPoints(pyramid_pts, rvec, tvec, K, dist_coeff)
        projected_pts = np.int32(projected_pts).reshape(-1, 2)

        # 바닥면: 피라미드의 베이스를 그리기 (파란색 선)
        cv.polylines(img, [projected_pts[:4]], isClosed=True, color=(255, 0, 0), thickness=2)
        # 각 바닥 모서리에서 꼭짓점까지 선 그리기 (초록색 선)
        for i in range(4):
            cv.line(img, tuple(projected_pts[i]), tuple(projected_pts[4]), color=(0, 255, 0), thickness=2)

        # 카메라 위치 출력: 회전 행렬과 평행 이동으로 카메라 중심 계산 (-R.T @ tvec)
        R, _ = cv.Rodrigues(rvec)
        cam_pos = (-R.T @ tvec).flatten()
        info = f'XYZ: [{cam_pos[0]:.3f} {cam_pos[1]:.3f} {cam_pos[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    
    # 결과 영상 표시 및 키 입력 처리
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27:  # ESC 키로 종료
        break

video.release()
cv.destroyAllWindows()
