# DGX Spark에서 Robocurve 과제 돌리기

지금 쓰는 서버는 x86 CPU에 GPU 8개입니다. 이 서버에서 하던 데모 녹화, Dream 실행, 제출을 NVIDIA DGX Spark(ARM CPU, GB10 칩)로 옮길 때 확인할 점과 시험 절차를 정리했습니다.

2026-11-20에 지금 서버에서 확인한 사실과, Spark에서 아직 시험하지 않은 부분을 구분해 적었습니다.

## 요약

| 항목 | 결론 |
|---|---|
| 모니터(디스플레이) | **필요 없음.** 지금 서버도 `DISPLAY`가 비어 있습니다. Isaac Sim이 화면 없이 GPU로 렌더링하고, 카메라 영상은 파일로 저장합니다. |
| CPU 구조 | **다름.** Spark는 arm64이고, 지금 쓰는 `yam-inspect-isaac:6.1.0` 이미지는 amd64입니다. 그대로 옮기면 실행되지 않으므로 Spark에서 이미지를 다시 빌드해야 합니다. |
| 기반 이미지 | `nvcr.io/nvidia/isaac-sim:6.1.0`은 amd64판과 arm64판이 모두 있습니다(`docker manifest inspect`로 확인). |
| GPU 수 | 1개입니다. 메모리 128 GB를 CPU와 함께 씁니다. 동시 실행은 1–2개로 제한합니다. |
| Dream·영상 업로드·PR 제출 | 네트워크로 하는 작업이라 기기와 관계없이 그대로 됩니다. 단 Dream 예산 장부는 복사하면 안 됩니다(5절 참고). |

## 1. 디스플레이가 필요 없는 이유

- 실행기(`.agents/skills/robocurve-demo-loop/scripts/robocurve.py run`)는 컨테이너를 `--livestream 0 --viz none`으로 실행합니다. 창을 띄우지 않습니다.
- `overview_cam.mp4` 같은 카메라 영상은 화면에 보여 주지 않고(offscreen) 컨테이너 안에서 렌더링한 뒤, ffmpeg로 파일에 저장합니다.
- WebRTC 라이브스트림은 `--browser` 옵션을 줄 때만 켜집니다. 브라우저로 실행 과정을 지켜보는 선택 기능이고, 기본값은 꺼져 있습니다.

## 2. arm64 이미지 빌드

과제 폴더의 `runtime/build_image.py`로 빌드합니다. 이 스크립트는 두 단계로 동작합니다.

1. `runtime/prepare.py`를 실행합니다. `runtime/versions.json`에 고정된 커밋의 IsaacLab-Arena와 yam-arena 소스를 GitHub에서 받아 `tasks/<task>/.runtime/`에 풉니다.
2. `runtime/Dockerfile`로 `docker build`를 실행합니다.

```bash
# Spark에서 실행합니다. 먼저 NGC 로그인이 필요합니다: docker login nvcr.io
docker pull nvcr.io/nvidia/isaac-sim:6.1.0
docker image inspect nvcr.io/nvidia/isaac-sim:6.1.0 --format '{{.Architecture}}'   # arm64가 나오는지 확인
python3 tasks/<task>/runtime/build_image.py            # 기본 태그: yam-inspect-isaac:6.1.0
```

**빌드가 실패할 수 있는 곳 (Spark에서 아직 시험하지 않음)**

1. **`isaaclab.sh --install mimic,teleop`:** teleop 쪽 의존성에 x86 전용 휠이 있으면 실패할 수 있습니다.
2. **`uv pip install`:** `inspect-robots-dream`, `dreamlabs-client==1.0.14`, `inspect-robots-agent`에 arm64 휠이 없으면 소스에서 빌드해야 합니다. 빌드에 필요한 `build-essential`과 `cmake`는 Dockerfile에서 이미 설치합니다.
3. **라이브스트림 라이브러리 복사 단계** (`libNvStreamBase.so`, `omni.kit.livestream.webrtc-*/bin`):
   - arm64 이미지에서 경로가 다르거나 파일이 없으면 `COPY`가 실패합니다.
   - 라이브스트림은 선택 기능이라, 여기서 막히면 이 단계만 빼도 데모와 Dream 녹화에는 영향이 없습니다.
4. **`torch`·`numba` 확인 단계:** arm64용 CUDA 휠이 Isaac Sim 6.1의 파이썬 3.12와 맞는지 확인해야 합니다.

## 3. 같은 결과가 나오는지 확인

빌드가 끝나면, 이미 합격한 과제 하나로 지금 서버와 같은 점수가 나오는지 확인합니다.

```bash
ROBOCURVE_CONFIG=~/.robocurve/config.json \
python3 .agents/skills/robocurve-demo-loop/scripts/robocurve.py run zip-top-bag-seal --gpu 0 --policy scripted --episodes 1
```

- **기대 점수:** 1.0입니다. 지금 서버에서는 616 step에 성공합니다.
  - PhysX 계산은 GPU와 CPU 구조에 따라 미세하게 달라질 수 있어서, step 수가 조금 다른 것은 정상입니다.
  - 점수가 1.0보다 낮게 나오면 데모 동작(웨이포인트)의 여유가 부족하다는 뜻입니다.
- **영상:** `runs/<task>/<run>/outputs/*/*/episode-01/overview_cam.mp4`가 정상으로 재생되는지 봅니다.
- **제출 검사:** 제출 전 검사(`verify_recorded_source`)는 과제 파일의 해시만 비교합니다. 그래서 Spark에서 이미지를 새로 빌드해도 제출 절차는 그대로입니다.

## 4. 자원 운용

- **GPU:** `--gpu 0` 하나뿐입니다. 지금 서버의 "여유 메모리가 30 GB 이상인 GPU만 쓴다"는 규칙은 Spark에서 "동시에 띄우는 컨테이너는 1–2개"로 바꿉니다.
- **메모리:** CPU와 GPU가 128 GB를 함께 씁니다. 컨테이너마다 `--shm-size 8g`가 잡히고 Isaac Sim 자체도 메모리를 많이 씁니다. 첫 실행 때 `free -g`와 `nvidia-smi`로 실제 사용량을 재고, 그 결과로 동시 실행 수를 정합니다.
- **속도:** 지금 서버처럼 데모와 Dream을 여러 GPU에서 동시에 돌릴 수 없습니다. 과제마다 데모 1회와 Dream 1회를 차례로 돌리게 됩니다.

## 5. 인증 정보와 Dream 예산 장부

- **옮길 파일:** `~/.robocurve/config.json`, `~/.robocurve/robocurve.env`, `~/.robocurve/dream.env`, `$HOME/.gh-config`
  - 옮긴 뒤에도 `chmod 600`으로 소유자만 읽을 수 있게 유지합니다.
  - Git에는 넣지 않습니다.
- **옮기면 안 되는 파일:** `~/.robocurve/dream-budget`(계정의 Dream 예산 장부)
  - 두 기기에 복사본이 생기면 예산이 이중으로 계산됩니다.
  - 워크스페이스 규칙도 "기존 장부를 그대로 쓰고, 복사·초기화·새로 만들기를 하지 않는다"입니다.
- **운용 방법:** 다음 중 하나를 고릅니다.
  - **권장:** Dream은 계속 지금 서버에서만 실행하고, Spark는 데모(scripted) 녹화와 장면 작업에만 씁니다.
  - Spark로 완전히 옮기고, 지금 서버에서는 Dream을 더 이상 실행하지 않습니다. 이 경우 장부를 복사하는 것이 아니라 옮기는 것이 됩니다.

## 6. 준비물 확인 목록

- [ ] NVIDIA 드라이버와 NVIDIA Container Toolkit이 설치되어 있음. `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi`가 동작하면 됩니다.
- [ ] `docker login nvcr.io` 완료(NGC API 키 필요)
- [ ] 워크스페이스 복사: `.agents/`, `tasks/`, `assets/`
  - `runs/`는 필요한 것만 복사합니다.
  - `tasks/<task>/.runtime/`는 `prepare.py`가 다시 받으므로 복사하지 않아도 됩니다.
- [ ] arm64 이미지 빌드 성공
- [ ] 합격 과제 하나의 데모가 1.0
- [ ] Dream 장부 운용 방법 결정(5절)
