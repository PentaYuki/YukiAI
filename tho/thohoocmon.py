# thohoocmon.py
import logging

logger = logging.getLogger(__name__)

class EmotionHarmonyBridge:
    def __init__(self, attention_model, emotion_integrator):
        self.attention_model = attention_model
        self.emotion_integrator = emotion_integrator
        self.current_ampha = None
        
        # === TÍCH HỢP MỚI: HỆ SỐ TÍNH CÁCH ===
        # Hệ số này quyết định mức độ ảnh hưởng của tính cách lên phản ứng hormone
        self.personality_impact_factor = 1.5
        
        self.emotion_mapping = {
            'vui_ve': 'HY',
            'gian_du': 'no',
            'yeu_thuong': 'ai',
            'ghet_bo': 'O',
            'buon_ba': 'BI',
            'doc_ac': 'A',
            'hanh_phuc': 'lac',
            'so_hai': 'Cu',
            'cang_thang': 'no+Cu',
            'an_toan': 'ai+lac'
        }
        self.emotion_to_hormone = {
            'HY': {'dopamine': 0.3, 'endorphins': 0.2, 'serotonin': 0.1},
            'no': {'cortisol': 0.4, 'adrenaline': 0.3, 'norepinephrine': 0.2},
            'ai': {'oxytocin': 0.5, 'serotonin': 0.2},
            'O': {'cortisol': 0.3, 'norepinephrine': 0.2},
            'BI': {'cortisol': 0.4, 'serotonin': -0.3},
            'A': {'adrenaline': 0.5, 'cortisol': 0.4},
            'lac': {'serotonin': 0.6, 'dopamine': 0.3, 'oxytocin': 0.1},
            'Cu': {'cortisol': 0.5, 'adrenaline': 0.4, 'norepinephrine': 0.3}
        }
        logger.info("EmotionHarmonyBridge initialized with Personality Impact Factor.")

    def map_to_basic_emotions(self, emotion_state):
        """Ánh xạ trạng thái cảm xúc thành các cảm xúc cơ bản"""
        basic_emotions = {}
        logger.debug(f"Mapping emotional state: {emotion_state}")

        for emotion, intensity in emotion_state.items():
            if not isinstance(intensity, (int, float)) or intensity <= 0.01:
                continue

            emotion_code = self.emotion_mapping.get(emotion)
            if emotion_code:
                components = emotion_code.split('+')
                num_components = len(components)
                for component in components:
                    if component in self.emotion_to_hormone:
                        basic_emotions[component] = basic_emotions.get(component, 0) + (intensity / num_components)

        logger.debug(f"Mapped to basic emotions: {basic_emotions}")
        
        # Chuẩn hóa để tổng cường độ không vượt quá 1, giữ tỷ lệ
        total = sum(basic_emotions.values())
        if total > 1.0:
            for emotion in basic_emotions:
                basic_emotions[emotion] /= total
        return basic_emotions
    
    def update_from_attention_model(self):
        """Cập nhật ampha từ trạng thái cảm xúc của mô hình chú ý"""
        emotional_state = self.attention_model.current_emotional_state
        logger.debug(f"Emotional state received for ampha generation: {emotional_state}")

        basic_emotions = self.map_to_basic_emotions(emotional_state)

        for emotion, intensity in basic_emotions.items():
            self.emotion_integrator.update_emotion_intensity(emotion, intensity)

        if basic_emotions:
            # Lấy các cảm xúc có cường độ lớn hơn 0 để tạo ampha
            dominant_emotions = tuple(sorted([em for em, intensity in basic_emotions.items() if intensity > 0.05]))
            if dominant_emotions:
                self.current_ampha = self.emotion_integrator.create_ampha(*dominant_emotions)
    
    def apply_ampha_to_hormones(self):
        """
        Tạo ra thay đổi hormone từ ampha hiện tại.
        ĐÃ CẬP NHẬT: Tích hợp ảnh hưởng từ vector tính cách.
        """
        if not self.current_ampha:
            return {}
        delta_vector = self.attention_model.delta_generator.get_current_delta()
        hormone_changes = {}
        intensity = self.current_ampha['intensity']
        current_levels = self.attention_model.hormone_system.get_summary()

        # === LẤY BẢN ĐỒ TÍNH CÁCH ===
        # Lấy hồ sơ tính cách đã chuẩn hóa từ module tích hợp
        personality_profile = self.emotion_integrator.get_personality_profile()

        for component in self.current_ampha['components']:
            if component in self.emotion_to_hormone:
                # === TÍNH TOÁN HỆ SỐ KHUẾCH ĐẠI TỪ TÍNH CÁCH ===
                # Nếu thành phần cảm xúc này đã có trong "tính cách", nó sẽ được khuếch đại
                personality_boost = personality_profile.get(component, 0) * self.personality_impact_factor
                
                for hormone, base_effect in self.emotion_to_hormone[component].items():
                    # Áp dụng khuếch đại từ tính cách
                    amplified_effect = base_effect * (1 + personality_boost)
                    
                    # Logic hiệu suất giảm dần (Diminishing returns)
                    current_level = current_levels.get(hormone, 0.0)
                    diminishing_factor = max(0, 1 - (current_level / 2.0)**2) 
                    
                    # Thay đổi cuối cùng bị ảnh hưởng bởi cường độ, tính cách và mức hormone hiện tại
                    change = amplified_effect * intensity * diminishing_factor
                    hormone_changes[hormone] = hormone_changes.get(hormone, 0) + change
                    
                    if personality_boost > 0:
                        logger.debug(f"Personality boost on '{component}' -> {hormone}: {change:.3f} (Boost: {personality_boost:.2f})")

        for i, value in enumerate(delta_vector):
            # Ánh xạ chiều delta sang hormone
            hormone_map = {
                0: 'dopamine',
                1: 'GABA',
                2: 'serotonin',
                3: 'cortisol',
                4: 'oxytocin',
                5: 'glutamate',
                6: 'adrenaline',
                7: 'endorphins',
            }
            
            hormone_name = hormone_map.get(i)
            if hormone_name:
                # Kết hợp delta với ảnh hưởng từ ampha
                hormone_changes[hormone_name] = hormone_changes.get(hormone_name, 0) + value * 0.3
        
        return hormone_changes
    
    def step(self):
        """Thực hiện một bước hoàn chỉnh của cầu nối."""
        self.update_from_attention_model()
        self.emotion_integrator.decay_amphas()
        dominant_ampha = self.emotion_integrator.get_current_dominant_ampha()
        
        # Cập nhật ampha hiện tại của cầu nối để apply_ampha_to_hormones sử dụng
        self.current_ampha = dominant_ampha 
        
        if dominant_ampha:
            logger.debug(f"Harmony Bridge Step -> Dominant Ampha: {dominant_ampha['key']} (Intensity: {dominant_ampha['intensity']:.2f})")
        
        return dominant_ampha