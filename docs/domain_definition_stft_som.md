# Domain 정의: STFT 기반 SOM 입력 (LoRa Baseband Chirp)

## 0) 요약
- **신호 상태**: Passband → Baseband (Carrier 제거 완료)
- **입력 데이터 형태**: Raw waveform이 아니라 **STFT(Short-Time Fourier Transform) 기반 스펙트로그램**(이미지 또는 벡터)
- **핵심 의도**: LoRa chirp의 **기울기(slope) / 패턴**을 시각적 특징으로 변환하여 **위상(phase) 변화에 강건**한 표현을 만들고, SOM이 이를 **자기조직화(unsupervised)로 군집화/분류**하도록 함

---

## 1) 문제 정의 (Problem Statement)
LoRa 심볼은 시간에 따라 주파수가 선형으로 변하는 chirp이며, baseband에서의 신호는 위상 오프셋/채널 위상 변화가 있어도 본질적인 **chirp 패턴(시간-주파수 평면에서의 선형 구조)**은 비교적 안정적으로 유지된다.

따라서 SOM 입력을 Raw complex waveform 대신 STFT 기반 특징으로 정의하면,
- 위상(phase) 변화에 덜 민감
- chirp의 기울기/구조가 더 직접적으로 드러남
- SOM에서 “서로 비슷한 chirp 패턴”이 근접하게 매핑될 가능성이 높음

---

## 2) 신호 전처리 체인 (Signal Chain)

### 2.1 Baseband 가정
- 입력 심볼 $x[n]$는 **carrier 제거 후 baseband complex I/Q** 샘플이라고 가정
- 샘플링 주파수는 프로젝트에서 사용한 설정을 따른다 (예: $F_s \approx BW$ 또는 실제 구현에 맞는 $F_s$)

### 2.2 권장 추가 전처리(선택)
- **DC offset 제거**: $x \leftarrow x - \mathrm{mean}(x)$
- **에너지 정규화**: $x \leftarrow x / (\sqrt{\mathbb{E}[|x|^2]}+\epsilon)$
- (선택) 간단한 windowing 적용

---

## 3) 표현(Representation) 정의: STFT

### 3.1 STFT 정의
STFT는 다음과 같이 정의된다.

$$
X(m, \omega) = \sum_{n} x[n] w[n-mH] e^{-j\omega n}
$$

여기서 $w[\cdot]$는 window, $H$는 hop size.

### 3.2 SOM 입력으로 쓸 특징
기본적으로 **phase에 강건**하게 만들기 위해 복소 STFT를 직접 쓰기보다 다음 중 하나를 권장:

- **Magnitude spectrogram**: $|X(m, \omega)|$
- **Log-magnitude(dB)**: $\log(|X| + \epsilon)$ 또는 $20\log_{10}(|X| + \epsilon)$

> 참고: LoRa의 chirp 구조는 magnitude 스펙트로그램에서도 충분히 드러나며, 위상 오프셋(상수 위상 회전)에 덜 민감하다.

### 3.3 입력 형태: 이미지 vs 벡터
- **이미지 방식**: $S \in \mathbb{R}^{F\times T}$를 “스펙트로그램 이미지”로 취급
- **벡터 방식(권장: SOM에 바로 넣기 쉬움)**: $S$를 flatten하여
  $$\mathbf{s} = \mathrm{vec}(S) \in \mathbb{R}^{FT}$$

추가 옵션:
- 주파수 축 또는 시간 축으로 **downsample / pooling** 해서 차원 축소
- PCA/Random Projection으로 $FT \to d$ 축소 후 SOM 입력

---

## 4) STFT 파라미터 권장안 (초기값)
심볼 길이가 $N=2^{SF}\cdot OSF$ 정도(예: SF=9, OSF=1이면 N=512)일 때, 시작점으로 아래를 추천:

- `nperseg`(window length): 64 또는 128
- `noverlap`: `nperseg // 2` (예: 32 또는 64)
- `nfft`: 256 또는 512
- `window`: Hann

설계 원칙:
- window가 너무 짧으면 주파수 해상도가 부족
- window가 너무 길면 시간 축에서 chirp 기울기 추적이 둔감

---

## 5) 정규화/스케일링 (SOM 학습 안정화)
STFT 기반 벡터는 스케일/분산이 커질 수 있으므로 아래 중 하나를 필수로 권장:

1) **Feature-wise Z-score**
- 각 feature(=각 TF-bin)에 대해 평균 0, 분산 1

2) **Per-sample 정규화**
- 각 샘플 벡터 $\mathbf{s}$에 대해 $\mathbf{s} \leftarrow (\mathbf{s} - \mu_s)/(\sigma_s+\epsilon)$

3) **dB clipping (dynamic range 고정)**
- 예: $[-60, 0]$ dB 범위로 clip 후 정규화

---

## 6) 학습 목표(라벨 없이)와 해석

### 6.1 학습 목표
- SOM이 스펙트로그램에서 나타나는 chirp 패턴의 유사성을 기반으로
  - **군집(클러스터)**
  - **연속적인 구조(코드워드 순환성/거리 구조)**
  를 스스로 조직화하도록 한다.

### 6.2 기대되는 관찰
- 동일 SF에서 codeword에 따라 스펙트로그램 패턴이 미세하게 이동/회전/변형됨
- SOM의 U-matrix에서 경계가 형성되거나, winner map에서 유사 패턴이 뭉침

---

## 7) 실전 팁: LoRa에서는 “dechirp 후” TF가 더 선명할 수 있음
Raw chirp의 FFT/STFT는 평탄하게 보이는 경우가 많다. LoRa 수신에서는 흔히
- **dechirp(다운처프 곱)** 후 FFT를 취해 tone 피크를 얻는다.

STFT에서도 마찬가지로,
- `x_de[n] = x[n] * conj(downchirp[n])`
- 그 다음 STFT를 취하면 **피크/선 구조**가 더 두드러질 수 있다.

즉, 두 가지 도메인 중 택1 또는 둘 다 실험 권장:
- Domain A: Baseband raw chirp → STFT magnitude
- Domain B(추천): Baseband dechirped → STFT magnitude

---

## 8) 구현 스케치 (Python)
아래는 “벡터 입력”을 만드는 최소 예시이다.

```python
import numpy as np
from scipy.signal import stft

def stft_features(x, fs, nperseg=128, noverlap=64, nfft=512, eps=1e-12,
                  use_db=True, clip_db=(-60, 0)):
    # x: complex baseband (N,)
    f, t, Z = stft(x, fs=fs, window='hann', nperseg=nperseg,
                  noverlap=noverlap, nfft=nfft, return_onesided=False,
                  boundary=None, padded=False)

    mag = np.abs(Z)
    if use_db:
        S = 20*np.log10(mag + eps)
        if clip_db is not None:
            S = np.clip(S, clip_db[0], clip_db[1])
    else:
        S = mag

    # (F, T) -> vector
    v = S.reshape(-1)

    # optional: per-sample normalize
    v = (v - v.mean()) / (v.std() + 1e-8)
    return v
```

---

## 9) 평가 방법 (라벨이 있을 때/없을 때)
- 라벨 없음:
  - U-matrix 경계가 생기는지
  - BMU 분포가 연속/구조를 가지는지
  - quantization error / topographic error
- 라벨 있음(예: SF 또는 codeword):
  - winner map에서 label purity
  - 인접 label이 인접 BMU로 가는지(거리 분석)

---

## 10) 코멘트 (요청하신 방향에 대한 의견)
이 방향(“Baseband + STFT 입력”)은 LoRa처럼 **시간-주파수 패턴이 본질**인 신호에서 굉장히 합리적입니다.
특히 위상 변화/상수 위상 회전에 강건하게 만들고 싶다면, **complex waveform보다 magnitude 기반 TF 표현**이 SOM에 더 잘 맞는 경우가 많아요.

다만 raw chirp는 TF가 평탄하게 보일 수 있어서, 실험 효율 측면에서 **dechirp 후 STFT/FFT 도메인(B)**도 같이 비교하는 걸 강하게 추천합니다.
