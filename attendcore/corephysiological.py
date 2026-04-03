import logging
logger = logging.getLogger(__name__)
class PhysiologicalCenter:
    def __init__(self):
        # Trạng thái sinh lý ban đầu
        self.states = {
            'heart_rate': 120,    # Nhịp tim (bpm)
            'breathing_rate': 30, # Nhịp thở (nhịp/phút)
            'tremor_level': 0.0,  # Mức độ run (0-1)
            'sweating': 0.0       # Đổ mồ hôi (0-1)
        }
        
        # Ánh xạ hormone tới các hàm xử lý
        self.HORMONE_IMPACT = {
            'cortisol': self._handle_stress,
            'adrenaline': self._handle_arousal,
            'GABA': self._handle_inhibition,
            'oxytocin': self._handle_calm
        }

        # Ánh xạ các thành phần AMPA (cảm xúc) tới các hàm xử lý
        self.AMPA_IMPACT = {
            'HY': self._handle_joy,
            'no': self._handle_anger,
            'ai': self._handle_love,
            'O': self._handle_hate,
            'BI': self._handle_sadness,
            'A': self._handle_evil,
            'lac': self._handle_bliss,
            'Cu': self._handle_fear
        }
    
    def _handle_stress(self, level):
        """Xử lý tác động của Cortisol (căng thẳng)"""
        self.states['heart_rate'] += int(40 * level)
        self.states['breathing_rate'] += int(20 * level)
        self.states['tremor_level'] = min(1.0, 0.5 * level)
    
    def _handle_arousal(self, level):
        """Xử lý tác động của Adrenaline (kích thích/hưng phấn)"""
        self.states['heart_rate'] += int(30 * level)
        self.states['sweating'] = min(1.0, 0.7 * level)
    
    def _handle_inhibition(self, level):
        """Xử lý tác động của GABA (ức chế/làm dịu)"""
        if level > 1.2: # Chỉ tác động khi GABA ở mức cao
            self.states['heart_rate'] -= int(25 * (level - 1.2))
            self.states['breathing_rate'] -= int(15 * (level - 1.2))
    
    def _handle_calm(self, level):
        """Xử lý tác động của Oxytocin (bình tĩnh)"""
        self.states['heart_rate'] -= int(20 * level)
        self.states['tremor_level'] = max(0.0, self.states['tremor_level'] - 0.4 * level)

    def _handle_joy(self, intensity):
        """Xử lý tác động của cảm xúc 'HY' (Vui mừng)"""
        self.states['heart_rate'] -= int(10 * intensity)
        self.states['breathing_rate'] -= int(5 * intensity)
        self.states['sweating'] = max(0.0, self.states['sweating'] - 0.2 * intensity)
    
    def _handle_anger(self, intensity):
        """Xử lý tác động của cảm xúc 'no' (Tức giận)"""
        self.states['heart_rate'] += int(20 * intensity)
        self.states['sweating'] = min(1.0, self.states['sweating'] + 0.3 * intensity)

    def _handle_love(self, intensity):
        """Xử lý tác động của cảm xúc 'ai' (Yêu thương)"""
        self.states['heart_rate'] -= int(5 * intensity)
        self.states['tremor_level'] = max(0.0, self.states['tremor_level'] - 0.1 * intensity)
        self.states['sweating'] = max(0.0, self.states['sweating'] - 0.1 * intensity)

    def _handle_hate(self, intensity):
        """Xử lý tác động của cảm xúc 'O' (Ghét bỏ)"""
        self.states['heart_rate'] += int(15 * intensity)
        self.states['breathing_rate'] += int(10 * intensity)
        self.states['tremor_level'] = min(1.0, self.states['tremor_level'] + 0.2 * intensity)

    def _handle_sadness(self, intensity):
        """Xử lý tác động của cảm xúc 'BI' (Buồn bã)"""
        self.states['heart_rate'] -= int(5 * intensity)
        self.states['breathing_rate'] -= int(2 * intensity)
        self.states['tremor_level'] = min(1.0, self.states['tremor_level'] + 0.1 * intensity)
        
    def _handle_evil(self, intensity):
        """Xử lý tác động của cảm xúc 'A' (Ác độc)"""
        self.states['heart_rate'] += int(25 * intensity)
        self.states['tremor_level'] = min(1.0, self.states['tremor_level'] + 0.3 * intensity)
        self.states['sweating'] = min(1.0, self.states['sweating'] + 0.4 * intensity)

    def _handle_bliss(self, intensity):
        """Xử lý tác động của cảm xúc 'lac' (Hạnh phúc tột độ)"""
        self.states['heart_rate'] -= int(15 * intensity)
        self.states['breathing_rate'] -= int(8 * intensity)
        self.states['tremor_level'] = 0.0 # Hoàn toàn bình tĩnh
        self.states['sweating'] = 0.0 # Không đổ mồ hôi

    def _handle_fear(self, intensity):
        """Xử lý tác động của cảm xúc 'Cu' (Sợ hãi)"""
        self.states['heart_rate'] += int(30 * intensity)
        self.states['breathing_rate'] += int(25 * intensity)
        self.states['tremor_level'] = min(1.0, self.states['tremor_level'] + 0.4 * intensity)
        self.states['sweating'] = min(1.0, self.states['sweating'] + 0.5 * intensity)


    def update(self, hormone_levels, ampha=None):
        """
        Cập nhật trạng thái sinh lý dựa trên nồng độ hormone và tác động ampha.
        Args:
            hormone_levels (dict): Dictionary chứa nồng độ của các hormone.
            ampha (dict, optional): Dictionary chứa thông tin về tác động ampha,
                                     bao gồm 'components' (danh sách các loại cảm xúc)
                                     và 'intensity' (cường độ). Mặc định là None.
        Returns:
            dict: Trạng thái sinh lý đã cập nhật.
        """
        # 1. Reset về trạng thái cơ bản
        # Nhịp tim cơ bản có thể thay đổi tùy thuộc vào hormone Melatonin (ví dụ, thấp hơn khi ngủ)
        base_heart = 120 if hormone_levels.get('melatonin', 0) < 0.4 else 100
        self.states = {
            'heart_rate': base_heart,
            'breathing_rate': 30,
            'tremor_level': 0.0,
            'sweating': 0.0
        }
        
        # 2. Áp dụng tác động hormone
        # Duyệt qua các hormone đã định nghĩa và áp dụng hàm xử lý tương ứng.
        for hormone, handler in self.HORMONE_IMPACT.items():
            if hormone in hormone_levels:
                handler(hormone_levels[hormone])
        
        # 3. Áp dụng ảnh hưởng của ampha (cảm xúc)
        # Nếu có thông tin ampha, duyệt qua các thành phần cảm xúc và áp dụng tác động.
        if ampha and 'components' in ampha and 'intensity' in ampha:
            logger.debug(f"Applying ampha effect: {ampha['key']} (intensity: {ampha['intensity']:.2f})")
            for component in ampha['components']:
                if component in self.AMPA_IMPACT:
                    self.AMPA_IMPACT[component](ampha['intensity'])
        
        # 4. Áp dụng giới hạn sinh lý
        # Đảm bảo các trạng thái sinh lý nằm trong giới hạn hợp lý.
        self.states['heart_rate'] = max(90, min(180, self.states['heart_rate']))
        self.states['breathing_rate'] = max(20, min(60, self.states['breathing_rate']))
        self.states['tremor_level'] = max(0.0, min(1.0, self.states['tremor_level']))
        self.states['sweating'] = max(0.0, min(1.0, self.states['sweating']))
        logger.debug(f"Final physiological state: {self.states}")
        return self.states