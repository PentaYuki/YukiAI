# thoemotion.py
import time
import logging
from collections import defaultdict # Thêm defaultdict

logger = logging.getLogger(__name__)

class EmotionIntegrationModule:
    def __init__(self):
        # Biểu đồ cảm xúc cơ bản
        self.emotion_map = {
            'HY': 'hỷ',  # Vui
            'no': 'nộ',  # Giận
            'ai': 'ái',  # Yêu thương
            'O': 'ố',    # Ghét
            'BI': 'bi',  # Buồn
            'A': 'ác',   # Ác độc
            'lac': 'lạc',# Hạnh phúc
            'Cu': 'cụ'   # Sợ hãi
        }
        
        # Lưu trữ các ampha đã tạo
        self.amphas = {}
        # Lưu trữ các lớp kết hợp
        self.combination_layers = {}
        # Bản đồ cường độ cảm xúc
        self.emotion_intensities = {emotion: 0.0 for emotion in self.emotion_map.keys()}
        
        # === TÍCH HỢP MỚI: VECTOR TÍNH CÁCH ===
        # Lưu trữ các xu hướng cảm xúc dài hạn
        self.personality_vector = defaultdict(float)
        logger.info("EmotionIntegrationModule initialized with Personality Vector.")

    def update_emotion_intensity(self, emotion, intensity):
        """Cập nhật cường độ cảm xúc cơ bản"""
        if emotion in self.emotion_intensities:
            self.emotion_intensities[emotion] = max(0.0, min(1.0, intensity))
    
    def create_ampha(self, *emotions, intensity_factor=1.0):
        if not emotions: return None
        key = ''.join(sorted(emotions)) # Sắp xếp để đảm bảo key nhất quán
        
        # Cập nhật cường độ nếu ampha đã tồn tại
        if key in self.amphas:
            # Lấy cường độ từ các thành phần cảm xúc
            total_intensity = sum(self.emotion_intensities.get(em, 0) for em in emotions)
            new_intensity = (total_intensity / len(emotions)) * intensity_factor if emotions else 0
            # Cập nhật với giá trị lớn hơn để phản ánh trạng thái hiện tại
            self.amphas[key]['intensity'] = max(self.amphas[key]['intensity'], new_intensity)
            self.amphas[key]['last_updated'] = time.time()
            return self.amphas[key]
        
        # PHƯƠNG PHÁP LẬP PHƯƠNG: Kết hợp phi tuyến cường độ
        cubic_intensity = 0.0
        valid_emotions = [em for em in emotions if em in self.emotion_intensities]
        if not valid_emotions: return None

        for emotion in valid_emotions:
            cubic_intensity += self.emotion_intensities[emotion] ** 3
        
        if cubic_intensity > 0:
            # Chuẩn hóa theo số lượng cảm xúc
            cubic_intensity = (cubic_intensity / len(valid_emotions)) ** (1/3)
        
        ampha = {
            'key': key,
            'components': valid_emotions,
            'intensity': min(1.0, cubic_intensity * intensity_factor),
            'layer': 0,
            'created_at': time.time(),
            'last_updated': time.time()
        }
        self.amphas[key] = ampha
        return ampha

    def combine_amphas(self, ampha1, ampha2, intensity_factor=1.0):
        """Kết hợp hai ampha để tạo ampha mới"""
        # Sắp xếp key để nhất quán
        sorted_keys = sorted([ampha1['key'], ampha2['key']])
        new_key = f"{sorted_keys[0]}+{sorted_keys[1]}"
        
        if new_key in self.amphas:
            return self.amphas[new_key]
        
        combined_components = tuple(sorted(list(set(ampha1['components'] + ampha2['components']))))
        combined_intensity = min(1.0, (ampha1['intensity'] + ampha2['intensity']) / 2 * intensity_factor)
        
        new_ampha = {
            'key': new_key,
            'components': combined_components,
            'intensity': combined_intensity,
            'layer': max(ampha1['layer'], ampha2['layer']) + 1,
            'base_amphas': (ampha1['key'], ampha2['key']),
            'created_at': time.time()
        }
        
        layer_key = f"layer_{new_ampha['layer']}"
        if layer_key not in self.combination_layers:
            self.combination_layers[layer_key] = []
        self.combination_layers[layer_key].append(new_key)
        
        self.amphas[new_key] = new_ampha
        return new_ampha

    def get_ampha_intensity(self, ampha_key):
        """Lấy cường độ cảm xúc của ampha"""
        return self.amphas.get(ampha_key, {}).get('intensity', 0.0)

    def decay_amphas(self, decay_rate=0.95, time_unit=5.0):
        """Phân rã cường độ ampha theo thời gian"""
        current_time = time.time()
        for key in list(self.amphas.keys()):
            ampha = self.amphas[key]
            time_elapsed = current_time - ampha.get('last_updated', current_time)
            
            # Phân rã theo đơn vị thời gian (ví dụ: 5 giây)
            decay_factor = decay_rate ** (time_elapsed / time_unit)
            self.amphas[key]['intensity'] *= decay_factor
            
            if self.amphas[key]['intensity'] < 0.01:
                logger.debug(f"Ampha '{key}' decayed and removed.")
                del self.amphas[key]
    
    def get_current_dominant_ampha(self):
        """Lấy ampha có cường độ cao nhất hiện tại"""
        if not self.amphas:
            return None
        return max(self.amphas.values(), key=lambda x: x['intensity'])

    # === CÁC PHƯƠNG THỨC MỚI CHO TÍNH CÁCH ===

    def update_personality(self, ampha_key, intensity_delta=0.01):
        """Cập nhật personality vector dựa trên ampha trội.
        
        Args:
            ampha_key (str): Key của ampha đang trội.
            intensity_delta (float): Mức độ thay đổi, thường tỷ lệ với cường độ ampha.
        """
        # Tăng giá trị cho ampha tương ứng, thể hiện sự củng cố
        current_value = self.personality_vector.get(ampha_key, 0)
        self.personality_vector[ampha_key] = min(3.0, current_value + intensity_delta) # Có giới hạn trên
        
        # Giảm nhẹ các ampha khác để thể hiện sự cạnh tranh và thay đổi theo thời gian
        for key in list(self.personality_vector.keys()):
            if key != ampha_key:
                self.personality_vector[key] *= 0.999 # Tốc độ giảm rất chậm
        
        logger.debug(f"Personality vector updated for '{ampha_key}'. New value: {self.personality_vector[ampha_key]:.4f}")

    def get_personality_profile(self):
        """Trả về personality vector đã được chuẩn hóa (tổng bằng 1)."""
        total = sum(self.personality_vector.values())
        if total > 0:
            return {k: v / total for k, v in self.personality_vector.items()}
        return {}