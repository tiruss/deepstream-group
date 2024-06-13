# CustomsLab AI/ML-Backend Integration

## 소개

이 저장소는 **"AI기반 분산카메라 환경 우범여행자 식별추적 시스템 개발"** 과제에서 개발하는 영상분석 모델을 **AI/ML-Backend**인 [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)로 통합하기 위한 것입니다.

DeepStream에서 각 영상인식 모델의 추론과 관련된 상세 내용은 **DeepStream**의 [Gst-nvinfer](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html)를 참고하시기 바랍니다.


## 사용법

#### 주의 사항

- 한글 출력을 위해서는 `fonts-nanum`을 호스트에 설치하고, `docker-compose.yaml`을 통해 경로를 마운트해야 함

#### 도커 컨테이너 실행

> 이후의 모든 과정은 도커 컨테이너 실행 후에 진행

  ```bash
  docker compose up -d
  ```

#### 신규 모델 사용 방법

- YOLOv8 모델 변환
  ```bash
  # 패키지 설치
  pip3 install ultralytics onnx
  # 모델 변환 - 544 x 960 해상도 (height x width)
  python3 pth2onnx.py --weights my_awesome_model.pt --imgsz 544 960
  # 변환 결과는 my_awesome_model.onnx, labels.txt
  ```
- 생성된 `onnx` 파일과 `label` 파일을 적절한 이름으로 수정하고 경로를 변경
  - onnx 파일은 `./engines/my_awesome_model.onnx`
  - labels.txt 파일은 `./configs/my_awesome_model_labels.txt`
- 새로운 config 파일 작성
  - `./configs/config_infer_primary_person.txt`를 참고하여 새로운 파일을 생성하고 아래 항목을 수정
    ```yaml
    onnx-file=../engines/my_awesome_model.onnx  # ONNX 파일 경로
    model-engine-file=../engines/my_awesome_model.onnx_b16_gpu0_fp16.engine  # 생성될 engine 파일 경로
    infer-dims=3;544;960  # 입력 해상도 (CHW)
    labelfile-path=p.txt  # label 파일
    num-detected-classes=1  # detector의 class 수

    # 필요시 아래의 IoU, confidence threshold 수정
    [class-attrs-all]
    nms-iou-threshold=0.7  # IoU threshold
    pre-cluster-threshold=0.25  # Confidence threshold
    ```

#### 실행

```bash
python3 app.py --help
usage: app.py [-h] --uris URIS [URIS ...] --ids IDS [IDS ...] [--sink SINK] [--sync-always]

DeepStream 기반 우범여행자 추적 프로그램

optional arguments:
  -h, --help            show this help message and exit
  --uris URIS [URIS ...]
                        List of URIs. --uris uri1 uri2 ... uriN
  --ids IDS [IDS ...]   List of camera IDs. --ids id1 id2 ... idN
  --sink SINK           Sink type (1: FakeSink, 2: EGLSink, 3: FileSink, 4: RTSP). Default sink is 2.
  --sync-always         Sync always
```

## 영상인식 모델 개발 주의사항

이 저장소에 모델을 추가하거나 수정하기 위해서는 `Merge Request`를 보내야 합니다.

각 모델 개발자는 반드시 [GitHub 프로젝트에 기여하기](https://git-scm.com/book/ko/v2/GitHub-GitHub-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%EC%97%90-%EA%B8%B0%EC%97%AC%ED%95%98%EA%B8%B0)를 참고하여 아래와 같은 순서로 작업을 진행해야 합니다.

1. 자신의 GitLab repository로 fork (최초 1회)
1. 로컬 repository로 clone (최초 1회)
1. `master` 브랜치를 기반으로 토픽 브랜치 생성
   > 새로운 내용을 작성하기 위해서는 반드시 원본 repository의 업데이트 내역을 반영해야 함
1. 뭔가 수정해서 커밋 (commit)

- `ONNX` 파일을 비롯한 바이너리 파일은 Git LFS 사용

5. 자신의 GitLab repository로 push
1. 원본 repository (여기)에 Merge Request 생성
1. 토론하면서 수정사항이 발생하면 계속 커밋

- 원본 repository가 업데이트 될 경우, 아래 내용을 참고하여 `rebase`
  - [Fork 한 Repository 업데이트 하기](https://velog.io/@k904808/Fork-%ED%95%9C-Repository-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8-%ED%95%98%EA%B8%B0)

8. Maintainer가 Merge Request를 merge하고 close
