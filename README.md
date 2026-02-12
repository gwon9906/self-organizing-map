# Self-Organizing Map (SOM) 실험 모음

이 저장소는 **SOM(Self-Organizing Map)** 으로 (1) 이미지 데이터(MNIST)와 (2) LoRa baseband 심볼을 **시각화/군집화**하는 실험을 정리합니다.

핵심 포인트는 “입력 표현(도메인/특징)을 어떻게 정의하느냐”에 따라 SOM의 토폴로지/군집 결과가 크게 달라진다는 점입니다.

## 빠른 시작

1) 파이썬 환경 준비(예: conda)

```bash
pip install numpy matplotlib scipy scikit-learn
```

MNIST 데이터를 로드하려면 보통 아래 중 하나가 필요합니다.

- (권장) 로컬에서 바로 로드: `pip install tensorflow`
- (대안) OpenML fallback: 네트워크가 되는 환경 + `scikit-learn`(위 설치에 포함)

2) 노트북 실행

- MNIST(0/1) 가중치 시각화: [model-test/simple_example/mnistBinary.ipynb](model-test/simple_example/mnistBinary.ipynb)
- LoRa STFT-SOM 군집 분석: [model-test/lora_som_analysis.ipynb](model-test/lora_som_analysis.ipynb)

## 폴더 구조

- [model-test](model-test)
  - [som.py](model-test/som.py): SOM 구현
  - [lora_som_analysis.ipynb](model-test/lora_som_analysis.ipynb): **STFT(+dechirp) 기반 LoRa 심볼 SOM 파이프라인 + 프로토타입(가중치) 시각화**
  - [lora_som_baseband.ipynb](model-test/lora_som_baseband.ipynb): LoRa baseband 실험(초기/비교용)
  - [simple_example/mnist.ipynb](model-test/simple_example/mnist.ipynb): MNIST(0~9) 데모
  - [simple_example/mnistBinary.ipynb](model-test/simple_example/mnistBinary.ipynb): **MNIST(0/1) 분리 + 프로토타입(가중치) 타일 시각화**
- [utils](utils)
  - [LoRa.py](utils/LoRa.py): LoRa 심볼 생성 유틸
- [docs](docs)
  - [domain_definition_stft_som.md](docs/domain_definition_stft_som.md): STFT-SOM 도메인 정의/설계 메모

## 핵심 결과 (노트북 저장 출력 기반)

### 1) MNIST Binary(0/1): “가중치(프로토타입) 타일”로 군집 구조 확인

노트북: [model-test/simple_example/mnistBinary.ipynb](model-test/simple_example/mnistBinary.ipynb)

- 데이터: MNIST에서 0/1만 사용, 총 6000개(0:3000, 1:3000)
- 입력: 784차원 → PCA 32차원 (Explained variance sum ≈ 0.8490)
- 학습(노트북 출력): Quantization Error ≈ 8.6781, Topographic Error ≈ 0.0900

관찰 포인트:

- U-Matrix/Label map에서 0과 1이 넓게 분리되는 경향이 나타남
- **프로토타입(뉴런 가중치) 타일**을 28×28로 복원(PCA inverse)하면, 맵 위에서
  - ‘1’의 세로 획/기울기 변화
  - ‘0’의 고리/두께/타원율 변화
  같은 “형태의 연속적인 변화”가 확인되어 **군집 경계가 무엇을 기준으로 생기는지** 직관적으로 해석 가능

결과 이미지(노트북 출력에서 추출):

![MNIST Binary - SOM U-Matrix](docs/results/mnistBinary_out_001.png)

![MNIST Binary - SOM prototypes (tiled)](docs/results/mnistBinary_out_004.png)

### 2) LoRa: STFT(+dechirp) 기반 SOM으로 “시간-주파수 패턴” 군집화

노트북: [model-test/lora_som_analysis.ipynb](model-test/lora_som_analysis.ipynb)

설정(노트북 출력):

- SF=9, codeword 0..511(총 512개), clean(no noise)
- BW=125 kHz 가정, FS=BW
- 파이프라인: (옵션) dechirp → STFT magnitude(dB) → 고정 크기(64×32) → 벡터화(2048) → PCA 64 → SOM(24×24, hex)
- 학습(노트북 출력): Quantization Error ≈ 4.7429, Topographic Error ≈ 0.0703

관찰 포인트:

- 단일 샘플 STFT 비교에서
  - **dechirp 전**: chirp가 시간-주파수 평면에서 대각선 구조로 나타남
  - **dechirp 후**: 특정 주파수 bin에 “수평 톤”이 뚜렷하게 나타남
- BMU 프로토타입(가중치)을 STFT로 복원하면, dechirp 후 톤 패턴과 유사한 프로토타입이 선택되어
  **SOM이 실제로 TF 패턴을 기준으로 근접도를 형성**함을 확인 가능
- 전체 프로토타입 타일(STFT-SOM prototypes)을 보면, 맵 전반에 걸쳐 “톤 위치/강도”가 변화하는 프로토타입들이 배열되어
  **코드워드별 패턴 차이가 SOM 격자에 조직화**되는 양상을 관찰 가능

결과 이미지(노트북 출력에서 추출):

![LoRa STFT-SOM prototypes (tiled)](docs/results/lora_som_analysis_out_002.png)

![LoRa single-sample STFT (no-dechirp vs dechirp) + BMU prototype](docs/results/lora_som_analysis_out_003.png)

## 재현성 메모

- SOM은 초기값/셔플에 따라 결과가 달라질 수 있어 `random_seed` 고정을 권장합니다.
- 노트북 출력 그림은 실행 환경(라이브러리 버전, 폰트/백엔드)에 따라 색/레이아웃이 조금 달라질 수 있습니다.
