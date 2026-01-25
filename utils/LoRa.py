"""
LoRa (Long Range) 신호 생성 유틸리티
- torch 의존성 제거된 경량 버전
"""
import numpy as np
import numpy.matlib
from scipy.signal import chirp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import signal


class LoRa:
    def __init__(self, sf, bw, OSF=8):
        """
        sf : spreading factor
        bw : bandwidth (Hz)
        OSF: oversampling factor (fs = bw * OSF)
        """
        self.sf = sf
        self.bw = bw
        self.OSF = OSF
        self.fs = int(bw * OSF)   # 전체 시스템에서 쓸 샘플링 주파수

    def gen_symbol(self, code_word, down=False, Fs=None):
        sf = self.sf
        bw = self.bw

        # 기본 샘플링 주파수는 self.fs = bw * OSF
        if Fs is None or Fs <= 0:
            Fs = self.fs

        # 이론적 심볼 시간 Ts
        Ts = (2 ** sf) / bw

        # 심볼 하나에 해당하는 샘플 수
        Ns = int(round(Ts * Fs))

        # 시간 축
        t = np.arange(Ns) / Fs

        # 기준 upchirp (baseband, -BW/2 ~ +BW/2 sweep)
        f0 = -bw / 2.0
        f1 = +bw / 2.0
        k = (f1 - f0) / Ts

        phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
        up_chirp = np.exp(1j * phase)

        # 데이터 심볼 m을 주파수 쉬프트로 실어줌
        m = int(code_word) % (2 ** sf)
        # 톤 주파수: f_m = m/Ts
        f_m = m / Ts
        phase_m = 2 * np.pi * f_m * t
        data_tone = np.exp(1j * phase_m)

        if not down:
            s = up_chirp * data_tone   # upchirp + data
        else:
            # downchirp = conj(upchirp) * data
            down_chirp = np.conj(up_chirp)
            s = down_chirp * data_tone

        return s.astype(np.complex128)

    def gen_symbol_exp(self, code_word, down=False):
        sf = self.sf
        bw = self.bw

        f_offset = bw/(2**sf) * code_word
        t_fold = (2**sf - code_word) / bw
        T = 2**sf/bw
        t1 = np.arange(0, t_fold, 1/bw)
        t2 = np.arange(t_fold, (2**sf)/bw, 1/bw)

        x1 = np.exp(1j*2*np.pi*(bw/(2*T)*(t1**2) + (f_offset - bw/2)*t1))
        x2 = np.exp(1j*2*np.pi*(bw/(2*T)*(t2**2) + (f_offset - 3*bw/2)*t2))
        result = np.concatenate((x1,x2),axis=0)
        if down:
            result = np.conj(result)
        return result
    
    def get_fft(self, signal):
        sig_fft = np.fft.fft(signal)
        return sig_fft
    
    def get_fft_abs(self, signal):
        sig_fft = self.get_fft(signal)
        sig_fft_abs = np.abs(sig_fft)
        return sig_fft_abs

    def plot_spectrogram(self, signal, noverlap=None, nfft=None):
        if noverlap is None and nfft is None:
            noverlap = 2**self.sf // 8
            nfft = 2**self.sf // 4
        plt.figure(figsize=(8,8))
        plt.specgram(signal, NFFT=nfft, noverlap=noverlap, Fs=self.bw)
        plt.show()

    def plot_fft_total(self, signal):
        x = np.arange(len(signal))
        sig_fft = self.get_fft(signal)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.text(0.5, 0.04, 'Frequency index', ha='center')
        fig.text(0.08, 0.45, 'Magnitude', rotation='vertical')
        
        ax1.set_title('Real Part')
        ax1.scatter(x, sig_fft.real, c='#1e88e5', alpha=0.7)
        ax1.plot(x, sig_fft.real, c='red', linestyle='dashed', alpha=0.5)

        ax2.set_title('Imaginary Part')
        ax2.scatter(x, sig_fft.imag, c='#1e88e5', alpha=0.7)
        ax2.plot(x, sig_fft.imag, c='red', linestyle='dashed', alpha=0.5)

        plt.show()

    def awgn_iq(self, signal_, SNR_):
        """IQ 복소 신호에 AWGN 노이즈 추가"""
        sig_avg_pwr = np.mean(abs(signal_)**2)
        noise_avg_pwr = sig_avg_pwr / (10**(SNR_/10))
        noise_sim = (np.random.normal(0, np.sqrt(noise_avg_pwr/2), len(signal_)) + 
                    1j*np.random.normal(0, np.sqrt(noise_avg_pwr/2), len(signal_)))
        return signal_ + noise_sim
    
    def add_awgn_noise(self, signal, snr_db):
        """주어진 SNR(dB)에 맞게 AWGN 노이즈 추가"""
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        return signal + noise
    
    def calculate_snr_db(self, clean_signal, noisy_signal):
        signal_power = np.mean(np.abs(clean_signal)**2)
        noise_power = np.mean(np.abs(noisy_signal - clean_signal)**2)
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db
    
    def gen_downchirp(self, Fs=None):
        """기준 downchirp 생성 (dechirp용)"""
        return self.gen_symbol(0, down=True, Fs=Fs)
    
    def dechirp(self, symbol, Fs=None):
        """
        심볼에 downchirp를 곱해서 dechirp 수행
        결과: 각 심볼이 고유한 주파수 bin에 피크를 가지게 됨
        """
        downchirp = self.gen_downchirp(Fs=Fs)
        # 길이 맞추기
        min_len = min(len(symbol), len(downchirp))
        dechirped = symbol[:min_len] * downchirp[:min_len]
        return dechirped
    
    def dechirp_and_fft(self, symbol, Fs=None):
        """Dechirp 후 FFT 수행하여 주파수 도메인 결과 반환"""
        dechirped = self.dechirp(symbol, Fs=Fs)
        fft_result = np.fft.fft(dechirped)
        return fft_result
    
    def demodulate(self, symbol, Fs=None):
        """
        심볼 복조: dechirp + FFT + argmax로 심볼 인덱스 추정
        Returns: 추정된 code word
        """
        fft_result = self.dechirp_and_fft(symbol, Fs=Fs)
        estimated_symbol = np.argmax(np.abs(fft_result))
        return estimated_symbol
