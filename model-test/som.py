"""
Self-Organizing Map (SOM) Implementation
Kohonen의 Self-Organizing Map 알고리즘 구현
- 커스텀 거리 함수 지원 추가
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple, Optional, List, Callable


class SelfOrganizingMap:
    """
    Self-Organizing Map (Kohonen Map) 구현
    
    Parameters:
    -----------
    map_size : Tuple[int, int]
        SOM 그리드 크기 (rows, cols)
    input_dim : int
        입력 벡터의 차원
    sigma : float
        초기 이웃 반경
    learning_rate : float
        초기 학습률
    decay_function : str
        감쇠 함수 타입 ('exponential', 'linear')
    distance_fn : str or Callable
        거리 함수 ('euclidean', 'cosine', 'manhattan' 또는 커스텀 함수)
    random_seed : int, optional
        재현성을 위한 랜덤 시드
    """
    
    def __init__(
        self,
        map_size: Tuple[int, int],
        input_dim: int,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        decay_function: str = 'exponential',
        distance_fn: str = 'euclidean',
        random_seed: Optional[int] = None
    ):
        self.map_size = map_size
        self.input_dim = input_dim
        self.sigma_0 = sigma
        self.learning_rate_0 = learning_rate
        self.decay_function = decay_function
        
        # 거리 함수 설정
        if callable(distance_fn):
            self._distance_fn = distance_fn
        elif distance_fn == 'euclidean':
            self._distance_fn = self._euclidean_distance
        elif distance_fn == 'cosine':
            self._distance_fn = self._cosine_distance
        elif distance_fn == 'manhattan':
            self._distance_fn = self._manhattan_distance
        else:
            self._distance_fn = self._euclidean_distance
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 가중치 초기화 (map_size[0] x map_size[1] x input_dim)
        self.weights = np.random.randn(map_size[0], map_size[1], input_dim) * 0.1
        
        # 뉴런 좌표 미리 계산
        self._neuron_positions = self._create_neuron_positions()
        
        # 학습 이력
        self.quantization_errors = []
        self.topographic_errors = []
    
    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """유클리드 거리"""
        return np.linalg.norm(a - b)
    
    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """코사인 거리 (1 - cosine similarity)"""
        dot = np.dot(a.flatten(), b.flatten())
        norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
        return 1 - dot / norm
    
    @staticmethod
    def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
        """맨해튼 거리"""
        return np.sum(np.abs(a - b))
        
    def _create_neuron_positions(self) -> np.ndarray:
        """각 뉴런의 그리드 좌표 생성"""
        rows, cols = self.map_size
        positions = np.zeros((rows, cols, 2))
        for i in range(rows):
            for j in range(cols):
                positions[i, j] = [i, j]
        return positions
    
    def _decay(self, initial_value: float, iteration: int, max_iterations: int) -> float:
        """학습률과 이웃 반경의 감쇠 계산"""
        if self.decay_function == 'exponential':
            # 안전한 time constant 계산
            time_constant = max_iterations / max(np.log(initial_value + 1), 1.0)
            return initial_value * np.exp(-iteration / time_constant)
        else:  # linear
            return initial_value * (1 - iteration / max_iterations)
    
    def init_weights_from_data(self, data: np.ndarray):
        """데이터 기반 가중치 초기화 (PCA 또는 랜덤 샘플링)"""
        n_samples = data.shape[0]
        
        # 방법 1: 데이터에서 랜덤 샘플링
        indices = np.random.choice(n_samples, size=self.map_size[0] * self.map_size[1], replace=True)
        sampled = data[indices]
        
        # 약간의 노이즈 추가
        noise = np.random.randn(*sampled.shape) * 0.01
        sampled = sampled + noise
        
        self.weights = sampled.reshape(self.map_size[0], self.map_size[1], -1)
    
    def _find_bmu(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Best Matching Unit (BMU) 찾기
        입력 벡터 x와 가장 유사한 뉴런의 위치 반환
        """
        min_dist = float('inf')
        bmu_idx = (0, 0)
        
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                dist = self._distance_fn(x, self.weights[i, j])
                if dist < min_dist:
                    min_dist = dist
                    bmu_idx = (i, j)
        
        return bmu_idx
    
    def _find_bmu_fast(self, x: np.ndarray) -> Tuple[int, int]:
        """
        BMU 찾기 (유클리드 거리용 빠른 버전)
        """
        diff = self.weights - x
        distances = np.linalg.norm(diff, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx
    
    def _find_second_bmu(self, x: np.ndarray, bmu: Tuple[int, int]) -> Tuple[int, int]:
        """두 번째 BMU 찾기 (topographic error 계산용)"""
        diff = self.weights - x
        distances = np.linalg.norm(diff, axis=2)
        
        # BMU 제외
        distances[bmu[0], bmu[1]] = np.inf
        
        second_bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return second_bmu_idx
    
    def _neighborhood_function(
        self, 
        bmu: Tuple[int, int], 
        sigma: float
    ) -> np.ndarray:
        """
        이웃 함수 계산 (Gaussian)
        BMU로부터의 거리에 따른 가중치 반환
        """
        bmu_pos = np.array(bmu)
        
        # 각 뉴런과 BMU 사이의 그리드 거리
        diff = self._neuron_positions - bmu_pos
        distances_sq = np.sum(diff ** 2, axis=2)
        
        # Gaussian 이웃 함수
        h = np.exp(-distances_sq / (2 * sigma ** 2))
        return h
    
    def train(
        self, 
        data: np.ndarray, 
        num_iterations: int,
        verbose: bool = True,
        calc_errors: bool = True
    ):
        """
        SOM 학습
        
        Parameters:
        -----------
        data : np.ndarray
            학습 데이터 (n_samples, input_dim)
        num_iterations : int
            학습 반복 횟수
        verbose : bool
            진행 상황 출력 여부
        calc_errors : bool
            에러 계산 여부
        """
        n_samples = data.shape[0]
        
        for iteration in range(num_iterations):
            # 랜덤하게 샘플 선택
            idx = np.random.randint(0, n_samples)
            x = data[idx]
            
            # 현재 학습률과 이웃 반경 계산
            learning_rate = self._decay(self.learning_rate_0, iteration, num_iterations)
            sigma = self._decay(self.sigma_0, iteration, num_iterations)
            sigma = max(sigma, 0.5)  # 최소값 유지
            
            # BMU 찾기
            bmu = self._find_bmu(x)
            
            # 이웃 함수 계산
            h = self._neighborhood_function(bmu, sigma)
            
            # 가중치 업데이트
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    self.weights[i, j] += learning_rate * h[i, j] * (x - self.weights[i, j])
            
            # 에러 계산 및 출력
            if verbose and (iteration + 1) % (num_iterations // 10) == 0:
                if calc_errors:
                    qe = self.quantization_error(data)
                    te = self.topographic_error(data)
                    self.quantization_errors.append(qe)
                    self.topographic_errors.append(te)
                    print(f"Iteration {iteration + 1}/{num_iterations} - "
                          f"QE: {qe:.4f}, TE: {te:.4f}, "
                          f"LR: {learning_rate:.4f}, σ: {sigma:.4f}")
                else:
                    print(f"Iteration {iteration + 1}/{num_iterations} - "
                          f"LR: {learning_rate:.4f}, σ: {sigma:.4f}")
    
    def train_batch(
        self,
        data: np.ndarray,
        num_epochs: int,
        verbose: bool = True
    ):
        """
        Batch SOM 학습 (전체 데이터를 사용한 업데이트)
        
        Parameters:
        -----------
        data : np.ndarray
            학습 데이터 (n_samples, input_dim)
        num_epochs : int
            에포크 수
        verbose : bool
            진행 상황 출력 여부
        """
        n_samples = data.shape[0]
        
        for epoch in range(num_epochs):
            # 현재 학습률과 이웃 반경
            learning_rate = self._decay(self.learning_rate_0, epoch, num_epochs)
            sigma = self._decay(self.sigma_0, epoch, num_epochs)
            sigma = max(sigma, 0.5)
            
            # 각 뉴런에 대해 BMU인 샘플들 수집
            numerator = np.zeros_like(self.weights)
            denominator = np.zeros((self.map_size[0], self.map_size[1], 1))
            
            for x in data:
                bmu = self._find_bmu(x)
                h = self._neighborhood_function(bmu, sigma)
                
                for i in range(self.map_size[0]):
                    for j in range(self.map_size[1]):
                        numerator[i, j] += h[i, j] * x
                        denominator[i, j] += h[i, j]
            
            # 가중치 업데이트
            self.weights = numerator / (denominator + 1e-10)
            
            if verbose:
                qe = self.quantization_error(data)
                te = self.topographic_error(data)
                self.quantization_errors.append(qe)
                self.topographic_errors.append(te)
                print(f"Epoch {epoch + 1}/{num_epochs} - QE: {qe:.4f}, TE: {te:.4f}")
    
    def quantization_error(self, data: np.ndarray) -> float:
        """
        양자화 오류 계산
        각 데이터 포인트와 BMU 사이의 평균 거리
        """
        errors = []
        for x in data:
            bmu = self._find_bmu(x)
            error = np.linalg.norm(x - self.weights[bmu[0], bmu[1]])
            errors.append(error)
        return np.mean(errors)
    
    def topographic_error(self, data: np.ndarray) -> float:
        """
        위상 오류 계산
        BMU와 2nd BMU가 이웃이 아닌 비율
        """
        errors = 0
        for x in data:
            bmu = self._find_bmu(x)
            second_bmu = self._find_second_bmu(x, bmu)
            
            # BMU와 2nd BMU 사이의 맨해튼 거리
            dist = abs(bmu[0] - second_bmu[0]) + abs(bmu[1] - second_bmu[1])
            
            # 이웃이 아니면 (거리 > 1) 에러
            if dist > 1:
                errors += 1
        
        return errors / len(data)
    
    def predict(self, x: np.ndarray) -> Tuple[int, int]:
        """입력 벡터의 BMU 반환"""
        return self._find_bmu(x)
    
    def get_winner_map(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        각 뉴런에 매핑된 데이터 개수 또는 레이블 반환
        
        Parameters:
        -----------
        data : np.ndarray
            입력 데이터
        labels : np.ndarray, optional
            데이터 레이블 (있으면 가장 빈번한 레이블 반환)
        
        Returns:
        --------
        winner_map : np.ndarray
            각 뉴런의 활성화 카운트 또는 대표 레이블
        """
        if labels is None:
            winner_map = np.zeros(self.map_size)
            for x in data:
                bmu = self._find_bmu(x)
                winner_map[bmu[0], bmu[1]] += 1
        else:
            # 각 뉴런에 매핑된 레이블 수집
            label_map = [[[] for _ in range(self.map_size[1])] for _ in range(self.map_size[0])]
            for x, label in zip(data, labels):
                bmu = self._find_bmu(x)
                label_map[bmu[0]][bmu[1]].append(label)
            
            # 가장 빈번한 레이블
            winner_map = np.full(self.map_size, -1, dtype=int)
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    if label_map[i][j]:
                        winner_map[i, j] = max(set(label_map[i][j]), key=label_map[i][j].count)
        
        return winner_map
    
    def get_u_matrix(self) -> np.ndarray:
        """
        U-Matrix 계산
        각 뉴런과 이웃 뉴런들 사이의 평균 거리
        """
        u_matrix = np.zeros(self.map_size)
        
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                neighbors = []
                
                # 4방향 이웃
                if i > 0:
                    neighbors.append(self.weights[i-1, j])
                if i < self.map_size[0] - 1:
                    neighbors.append(self.weights[i+1, j])
                if j > 0:
                    neighbors.append(self.weights[i, j-1])
                if j < self.map_size[1] - 1:
                    neighbors.append(self.weights[i, j+1])
                
                # 평균 거리 계산
                if neighbors:
                    distances = [np.linalg.norm(self.weights[i, j] - n) for n in neighbors]
                    u_matrix[i, j] = np.mean(distances)
        
        return u_matrix
    
    def plot_u_matrix(self, figsize: Tuple[int, int] = (10, 8), cmap: str = 'viridis'):
        """U-Matrix 시각화"""
        u_matrix = self.get_u_matrix()
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(u_matrix, cmap=cmap, interpolation='nearest')
        ax.set_title('U-Matrix (Unified Distance Matrix)')
        ax.set_xlabel('SOM Column')
        ax.set_ylabel('SOM Row')
        plt.colorbar(im, ax=ax, label='Average Distance to Neighbors')
        plt.tight_layout()
        return fig, ax
    
    def plot_component_planes(
        self, 
        component_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10),
        cmap: str = 'coolwarm'
    ):
        """
        각 입력 차원에 대한 컴포넌트 플레인 시각화
        """
        n_components = self.input_dim
        n_cols = min(4, n_components)
        n_rows = (n_components + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes)
        
        if component_names is None:
            component_names = [f'Component {i}' for i in range(n_components)]
        
        for idx in range(n_components):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            component = self.weights[:, :, idx]
            im = ax.imshow(component, cmap=cmap, interpolation='nearest')
            ax.set_title(component_names[idx])
            plt.colorbar(im, ax=ax)
        
        # 빈 subplot 숨기기
        for idx in range(n_components, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig, axes
    
    def plot_winner_map(
        self, 
        data: np.ndarray, 
        labels: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'tab20'
    ):
        """Winner map 시각화"""
        winner_map = self.get_winner_map(data, labels)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if labels is None:
            im = ax.imshow(winner_map, cmap='viridis', interpolation='nearest')
            ax.set_title('Hit Map (Activation Count)')
            plt.colorbar(im, ax=ax, label='Number of Data Points')
        else:
            im = ax.imshow(winner_map, cmap=cmap, interpolation='nearest')
            ax.set_title('Label Map (Most Frequent Label per Neuron)')
            plt.colorbar(im, ax=ax, label='Label')
        
        ax.set_xlabel('SOM Column')
        ax.set_ylabel('SOM Row')
        plt.tight_layout()
        return fig, ax
    
    def plot_data_on_map(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        figsize: Tuple[int, int] = (12, 10),
        alpha: float = 0.6,
        cmap: str = 'tab20'
    ):
        """
        데이터를 SOM 맵 위에 scatter plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(unique_labels)))
        
        # U-Matrix를 배경으로
        u_matrix = self.get_u_matrix()
        ax.imshow(u_matrix, cmap='gray_r', alpha=0.3, interpolation='nearest')
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            positions = []
            for x in data[mask]:
                bmu = self._find_bmu(x)
                # 약간의 jitter 추가
                jitter = np.random.uniform(-0.3, 0.3, 2)
                positions.append([bmu[1] + jitter[0], bmu[0] + jitter[1]])
            
            positions = np.array(positions)
            ax.scatter(positions[:, 0], positions[:, 1], 
                      c=[color], label=f'Symbol {label}', alpha=alpha, s=50)
        
        ax.set_xlim(-0.5, self.map_size[1] - 0.5)
        ax.set_ylim(self.map_size[0] - 0.5, -0.5)
        ax.set_title('Data Points on SOM')
        ax.set_xlabel('SOM Column')
        ax.set_ylabel('SOM Row')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.tight_layout()
        return fig, ax
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 4)):
        """학습 히스토리 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        if self.quantization_errors:
            axes[0].plot(self.quantization_errors, 'b-o')
            axes[0].set_title('Quantization Error')
            axes[0].set_xlabel('Checkpoint')
            axes[0].set_ylabel('Error')
            axes[0].grid(True)
        
        if self.topographic_errors:
            axes[1].plot(self.topographic_errors, 'r-o')
            axes[1].set_title('Topographic Error')
            axes[1].set_xlabel('Checkpoint')
            axes[1].set_ylabel('Error')
            axes[1].grid(True)
        
        plt.tight_layout()
        return fig, axes


class ComplexSOM(SelfOrganizingMap):
    def __init__(
        self,
        map_size: Tuple[int, int],
        input_dim: int,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        decay_function: str = 'exponential',
        random_seed: Optional[int] = None,
        use_magnitude_phase: bool = False,
        distance_fn: str = 'euclidean',  # 옵션으로 추가 추천
    ):
        super().__init__(
            map_size=map_size,
            input_dim=input_dim * 2,
            sigma=sigma,
            learning_rate=learning_rate,
            decay_function=decay_function,
            distance_fn=distance_fn,
            random_seed=random_seed
        )
        self.complex_input_dim = input_dim
        self.use_magnitude_phase = use_magnitude_phase

    
    def _complex_to_real(self, x: np.ndarray) -> np.ndarray:
        """복소수를 실수 벡터로 변환"""
        if self.use_magnitude_phase:
            magnitude = np.abs(x)
            phase = np.angle(x)
            return np.concatenate([magnitude, phase])
        else:
            return np.concatenate([x.real, x.imag])
    
    def _real_to_complex(self, x: np.ndarray) -> np.ndarray:
        """실수 벡터를 복소수로 변환"""
        mid = len(x) // 2
        if self.use_magnitude_phase:
            magnitude = x[:mid]
            phase = x[mid:]
            return magnitude * np.exp(1j * phase)
        else:
            return x[:mid] + 1j * x[mid:]
    
    def train_complex(
        self,
        data: np.ndarray,
        num_iterations: int,
        verbose: bool = True
    ):
        """복소수 데이터로 학습"""
        # 복소수를 실수로 변환
        real_data = np.array([self._complex_to_real(x) for x in data])
        self.train(real_data, num_iterations, verbose)
    
    def predict_complex(self, x: np.ndarray) -> Tuple[int, int]:
        """복소수 입력의 BMU 반환"""
        real_x = self._complex_to_real(x)
        return self._find_bmu(real_x)
    
    def get_complex_weights(self) -> np.ndarray:
        """가중치를 복소수로 변환하여 반환"""
        complex_weights = np.zeros((self.map_size[0], self.map_size[1], 
                                    self.complex_input_dim), dtype=complex)
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                complex_weights[i, j] = self._real_to_complex(self.weights[i, j])
        return complex_weights
