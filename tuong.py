import numpy as np
import time
from collections import defaultdict
from attendcore.corehocmon import ATTENTION_HORMONE_MAP, ATTENTION_STATUS_HORMONES ,  SENSOR_WEIGHTS, SAFETY_THRESHOLDS, SMELL_COMPOUNDS, TASTE_COMPOUNDS, SENSITIVE_AREAS
from attendcore.corehocmon2 import HormoneDynamics
from memory.expAttend import ExperienceEncoder ,PreferenceMemorySystem
from attendcore.corephysiological import PhysiologicalCenter
from sac.Vision import VisionProcessor
from sac.Audio import AudioProcessor
from tho.thoemotion import EmotionIntegrationModule
from thuc import ThucConsciousnessModule , RewardEvaluator
from tho.thohoocmon import EmotionHarmonyBridge
from attendcore.coredelta import DeltaGenerator
import torch
import torch.nn.functional as F
import math 
import logging
# --- Thiết lập Logging ---
# Thêm dòng này vào đầu tệp để cấu hình logging cơ bản
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class SensorProcessor:
    @staticmethod
    def process_vision(camera, thresholds, age_months):
        """Tính điểm thị giác với xử lý lỗi đầu vào"""
        logger.debug(f"Processing vision. Received thresholds type: {type(thresholds)}, value: {thresholds}")        
        score = 0.0
        attention_types = set()
        
        # 1. Kiểm tra flicker (chớp tắt) với giá trị mặc định
        # Dòng này sẽ không còn gây lỗi sau khi sửa ở trên
        max_flicker = thresholds.get('max_flicker', 30) 
        if camera.get('flicker_freq', 0) > max_flicker:
            score -= 0.4
            attention_types.add('overstimulated_by_flicker')
        
        # 2. Độ rõ nét và tương phản với giá trị mặc định
        clarity = camera.get('clarity', 0.5)
        min_clarity = thresholds.get('min_clarity', 0.4)
        if clarity > min_clarity:
            score += clarity * 0.25
        
        contrast = camera.get('contrast', 0.5)
        min_contrast = thresholds.get('min_contrast', 0.3)
        if contrast > min_contrast:
            score += contrast * 0.15
        
        # 3. Màu sắc: Ưu tiên màu đỏ và bão hòa cao
        saturation = camera.get('saturation', 0.5)
        hue_R = camera.get('hue_R', 0.0)
        if hue_R > 0.7 and saturation > 0.8:
            score += 0.15 * (1 + saturation)
            attention_types.add('bright_color_interest')
        elif hue_R > camera.get('hue_G', 0.0) and hue_R > camera.get('hue_B', 0.0) and saturation > 0.6:
            score += 0.1 * (1 + saturation)
            attention_types.add('bright_color_interest')
        
        # 4. Chuyển động (quan trọng với trẻ >3 tháng)
        motion = camera.get('motion', 0.0)
        if age_months > 3 and motion > 0.7:
            score += motion * 0.25
            attention_types.add('motion_tracking')
        
        # 5. Kích cỡ và khoảng cách (quan trọng với trẻ 6-24 tháng)
        if age_months >= 6:
            size = camera.get('object_size', 0.0)
            distance = camera.get('object_distance', 1.0)
            size_score = min(1.0, size * 2)
            dist_score = 1.0 - min(1.0, abs(distance - 0.75) * 2)
            score += (size_score * dist_score) * 0.3
            if size > 0.3 and 0.5 <= distance <= 1.0:
                score += 0.2
            elif size > 0.1:
                score += 0.1
        
        # 6. Nhận diện khuôn mặt
        if camera.get('face_detected', False):
            attention_types.add('face_focused')
            score += 0.2

        # 7. Xử lý các đối tượng được theo dõi
        tracked_objects = camera.get('tracked_objects', [])
        for obj in tracked_objects:
            stability = obj.get('stability', 0.0)
            if stability > 0.7:
                score += 0.3 * stability
                
            trajectory = obj.get('trajectory', [])
            if len(trajectory) > 1:
                last_pos = trajectory[-1]
                prev_pos = trajectory[-2]
                movement = math.sqrt((last_pos[0]-prev_pos[0])**2 + (last_pos[1]-prev_pos[1])**2)
                if movement > 5:
                    score += 0.2
                    attention_types.add('moving_object_tracking')
        
        # 8. Xử lý các proto-objects
        proto_objects = camera.get('proto_objects', [])
        for proto in proto_objects:
            complexity = proto.get('complexity', 0.0)
            score += min(0.4, complexity * 0.1)
            attention_types.add('complex_shape_interest')
        
        # 9. Xử lý các quan hệ không gian
        spatial_relations = camera.get('spatial_relations', {})
        for relations in spatial_relations.values():
            for rel_data in relations.values():
                relation = rel_data.get('relation', '')
                if relation == 'contains':
                    score += 0.15
                    attention_types.add('spatial_relation_containment')
                elif relation == 'parallel':
                    score += 0.1
                    attention_types.add('spatial_relation_parallel')
        # Thêm điểm cơ sở nếu có bất kỳ đối tượng nào
        if camera.get('geometric_features') or camera.get('tracked_objects'):
            score += 0.15  # Điểm cơ sở cho việc có vật thể trong tầm nhìn
            attention_types.add('visual_stimulus_present')                    
        
        return min(1.0, max(-0.5, score)), attention_types
class BabyAttentionModel:
    def __init__(
        self,
        age_months,
        hormone_system,
        memory_system,
        vision_processor,
        audio_processor,
        config,
        physio_center=None
    ):
        """
        Khởi tạo mô hình chú ý của bé, thiết lập các thuộc tính,
        cấu hình và các hệ thống phụ thuộc một cách đầy đủ và an toàn.
        """
        # === 1. Thuộc tính cơ bản & Trạng thái ===
        self.age_months = min(int(age_months), 23)  # Đảm bảo là số nguyên và trong khoảng 0-23
        self.age_group = self._get_age_group()
        self.medical_conditions = {
            'is_hungry': False,
            'is_in_pain': False,
            'is_sleeping': False,
            'is_tired': False,
            'is_sick': False
        }
        self.history = defaultdict(list)
        self.last_update_time = time.time()

        # === 2. Thiết lập Logging ===
        self.logger = logging.getLogger('BabyAttentionModel')
        self.logger.info(f"Initializing BabyAttentionModel for an infant of {self.age_months} months.")

        # === 3. Tải cấu hình một cách an toàn ===
        try:
            self.SAFETY_THRESHOLDS = config['SAFETY_THRESHOLDS']
            self.SENSOR_WEIGHTS = config['SENSOR_WEIGHTS']
            self.SMELL_COMPOUNDS = config['SMELL_COMPOUNDS']
            self.TASTE_COMPOUNDS = config['TASTE_COMPOUNDS']
            self.SENSITIVE_AREAS = config['SENSITIVE_AREAS']
            self.ATTENTION_TYPE_HORMONES = config['ATTENTION_HORMONE_MAP']
            self.ATTENTION_STATUS_HORMONES = config['ATTENTION_STATUS_HORMONES']
            # Dùng .get() cho tham số không bắt buộc để an toàn hơn
            self.STATUS_IMPACT_FACTOR = config.get('STATUS_IMPACT_FACTOR', 1.2)
            self.logger.info("Configuration loaded successfully.")
        except KeyError as e:
            self.logger.error(f"Missing critical key in config dictionary: {e}")
            raise ValueError(f"Configuration is missing a required key: {e}") from e

        # === 4. Dependency Injection (Gán các hệ thống phụ thuộc) ===
        # Kiểm tra các hệ thống bắt buộc phải có
        if not all([hormone_system, memory_system, vision_processor]):
            raise ValueError("Hormone, memory, and vision systems must be provided and not be None.")
        
        self.hormone_system = hormone_system
        self.memory_system = memory_system
        self.vision_processor = vision_processor
        self.audio_processor = audio_processor  # Có thể là None nếu không khởi tạo được
        self.physio_center = physio_center if physio_center is not None else PhysiologicalCenter()
        self.delta_generator = DeltaGenerator()  # <--- THÊM DÒNG NÀY
        self.emotion_integrator = EmotionIntegrationModule()
        self.harmony_bridge = EmotionHarmonyBridge(self, self.emotion_integrator)
        self.current_ampha = None
        self.preference_memory = PreferenceMemorySystem()  # Thêm hệ thống sở thích
        self.desired_ampha = None  # Ampha mong muốn hiện tại        
        self.amphas = self.emotion_integrator.amphas # Tham chiếu đến amphas trong bộ tích hợp
        self.personality_traits = defaultdict(float)
        # === THÊM MODULE THỨC ===
        self.consciousness = ThucConsciousnessModule(
            input_dim=128,  # Tổng hợp nội tại + môi trường + bộ nhớ
            hidden_dim=64,
            active_intent_dim=8,     # VD: KHÁM_PHÁ, TRÁNH, TÌM_AN_ỦI,...
            mental_disposition_dim=5 # VD: TÌM_VUI, TỰ_VỆ, TRÁNH_CHÁN,...
        )
        
        # Bộ đánh giá phần thưởng
        self.reward_evaluator = RewardEvaluator(
            short_term_weights={
                'dopamine': 1.0, 
                'oxytocin': 0.8,
                'cortisol': -1.0
            },
            long_term_weights={
                'oxytocin': 0.5,
                'serotonin': 0.5
            })      
        self.logger.info("All dependent systems have been injected and model is ready.")
    def _get_age_group(self):
        if self.age_months <= 3: return 1
        if self.age_months <= 6: return 2
        return 3
    
    def _get_sensor_weight(self, sensor_name):
        """Lấy trọng số sensor theo tháng tuổi"""
        return self.SENSOR_WEIGHTS[sensor_name][self.age_months]
    
    def _get_safety_threshold(self, category):
        """Lấy ngưỡng an toàn theo nhóm tuổi"""
        return self.SAFETY_THRESHOLDS[category][self.age_group]
    
    def update_medical_conditions(self, **conditions):
        """Cập nhật tình trạng y khoa"""
        for key, value in conditions.items():
            if key in self.medical_conditions:
                self.medical_conditions[key] = value
                self.logger.info(f"Medical condition updated: {key} = {value}")
    
    def _apply_gaba_inhibition(self, sensory_input):
        """Áp dụng hiệu ứng ức chế của GABA lên đầu vào cảm biến"""
        if 'GABA' not in self.hormone_system.levels:
            return sensory_input
        
        gaba_level = self.hormone_system.levels['GABA']
        # Chỉ áp dụng ức chế nếu GABA ở mức đáng kể
        if gaba_level < 0.8:    
            return sensory_input

        inhibition_factor = 1.0 - min(0.5, (gaba_level - 0.8) / 2.0)
        self.logger.debug(f"Applying GABA inhibition with factor: {inhibition_factor:.2f}")
        
        # Danh sách các thuộc tính cần giảm cường độ
        visual_attributes = [
            'clarity', 'saturation', 'motion', 
            'hue_R', 'hue_G', 'hue_B',
            'brightness', 'intensity', 'luminance'
        ]
        
        audio_attributes = ['volume_dB', 'complexity', 'intensity']
        
        # Áp dụng cho thị giác
        if 'camera' in sensory_input and sensory_input['camera'] is not None:
            self.logger.debug(f"Camera input for attention calculation. Type: {type(sensory_input['camera'])}")
        else:
            self.logger.warning("No 'camera' key in sensory_input for this step.")            
            for attr in visual_attributes:
                # Kiểm tra thêm xem thuộc tính có trong dict camera không
                if attr in sensory_input['camera']:
                    sensory_input['camera'][attr] *= inhibition_factor
        
        # === KIỂM TRA AN TOÀN CHO THÍNH GIÁC ===
        # Chỉ xử lý thính giác NẾU key 'mic' tồn tại và giá trị của nó không phải là None
        if 'mic' in sensory_input and sensory_input['mic'] is not None:
            for attr in audio_attributes:
                # Kiểm tra thêm xem thuộc tính có trong dict mic không
                if attr in sensory_input['mic']:
                    sensory_input['mic'][attr] *= inhibition_factor
        else:
            self.logger.debug("Skipping GABA inhibition for audio: 'mic' key not found or not a dict.")
    
        return sensory_input      
    
    def _apply_gaba_feedback(self, attention_types, scores):
        """Áp dụng phản hồi sinh lý của GABA lên hệ thống chú ý"""
        if 'GABA' not in self.hormone_system.levels:
            return attention_types, scores
        
        gaba_level = self.hormone_system.levels['GABA']
        # GABA cao → tăng ngưỡng kích thích
        if gaba_level > 1.0:
            if 'hearing' in scores:
                scores['hearing'] *= 0.7
            if 'vision' in scores:
                scores['vision'] *= 0.8
            attention_types.add('gaba_inhibition')
        
        return attention_types, scores
    
    def _apply_gaba_attention_decay(self, attention_score, attention_types):
        """Áp dụng suy giảm chú ý khi GABA cao"""
        if 'GABA' not in self.hormone_system.levels:
            return attention_score
        
        gaba_level = self.hormone_system.levels['GABA']
        if gaba_level > 1.1:  # Ngưỡng GABA cao
            decay_factor = 1.0 - min(0.5, (gaba_level - 1.1) * 0.8)
            
            # Loại trừ các attention types đặc biệt
            protected_types = {
                'basic_needs_required', 
                'pain_response',
                'emergency_attention'
            }
            if not protected_types.intersection(attention_types):
                attention_score *= decay_factor
                attention_types.add('gaba_attention_decay')
        
        return attention_score    
    
    def calculate_attention(self, sensory_input):
        """Tính toán điểm chú ý và xác định loại attention"""
        attention_types = set()
        if self.medical_conditions['is_sick']:
            return 0.05, "unattend (sick)", {}, {'sickness'}
        
        vision_thresholds = self._get_safety_threshold('vision') 
        self.logger.debug(f"Using vision thresholds for age group {self.age_group}: {vision_thresholds}")

        vision_score, vision_attention = SensorProcessor.process_vision(
            sensory_input['camera'], 
            vision_thresholds, # <--- SỬA Ở ĐÂY: Truyền dictionary đã được chọn
            self.age_months                   # Truyền thông tin tuổi
        )

        hearing_score, hearing_attention = self._calculate_hearing_score(sensory_input['mic'])
        smell_score, smell_attention = self._calculate_smell_score(sensory_input['smell'])
        taste_score, taste_attention = self._calculate_taste_score(sensory_input['taste'])
        touch_score, touch_attention = self._calculate_touch_score(sensory_input['force'])
        temp_score, temp_attention = self._calculate_temp_score(sensory_input['temperature_C'])
        
        scores = {
            'vision': vision_score,
            'hearing': hearing_score,
            'smell': smell_score,
            'taste': taste_score,
            'touch': touch_score,
            'temperature': temp_score
        }
        self.logger.debug(f"Individual sensor scores: { {k: f'{v:.2f}' for k, v in scores.items()} }")
        
        # Tổng hợp tất cả attention types
        attention_types.update(vision_attention)
        attention_types.update(hearing_attention)
        attention_types.update(smell_attention)
        attention_types.update(taste_attention)
        attention_types.update(touch_attention)
        attention_types.update(temp_attention)
        
        # Tính điểm tổng có trọng số
        total_score = 0.0
        total_weight = 0.0
        for sensor, score in scores.items():
            weight = self._get_sensor_weight(sensor)
            total_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            normalized_score = total_score / total_weight 
        else:
            normalized_score = 0.0
        # Xử lý điều kiện y tế trước
        if self.medical_conditions['is_sleeping']:
            attention_types.add('sleeping')
            return 0.0, "unattend (sleeping)", {}, set(attention_types)
        
        if self.medical_conditions['is_hungry'] or self.medical_conditions['is_in_pain']:
            attention_types.add('basic_needs_required')
            return 0.1, "unattend (basic needs)", {}, set(attention_types)
        if self.desired_ampha:
            preference_boost = self._get_preference_boost(sensory_input)
            total_score *= preference_boost 
            if preference_boost > 1.0:
                self.logger.debug(f"Applying preference boost: {preference_boost:.2f}")        
        # Áp dụng suy giảm chú ý do GABA
        normalized_score = self._apply_gaba_attention_decay(normalized_score, attention_types)
        
        # Xác định trạng thái chú ý tổng
        if normalized_score > 0.7:
            status = "high attend"
        elif normalized_score > 0.5:
            status = "moderate attend"
        elif normalized_score > 0.3:
            status = "neutral"
        elif normalized_score > 0.1:
            status = "unattend"
        else:
            status = "strong unattend"
        
        # Trả về kết quả
        return normalized_score, status, scores, set(attention_types)
    def _get_preference_boost(self, sensory_input):
        """Tăng cường điểm chú ý cho kích thích phù hợp sở thích"""
        current_pattern = self._extract_sensory_pattern(sensory_input)
        preferred_stimuli = self.preference_memory.get_preferred_stimuli(self.desired_ampha)
        
        for stimulus in preferred_stimuli:
            if self._similarity(current_pattern, stimulus) > 0.7:
                return 1.5  # Tăng 50% điểm chú ý
        return 1.0    
    def _check_emotional_states(self, sensory_input, attention_types):
        """Kiểm tra các trạng thái cảm xúc đặc biệt"""
        # ... [giữ nguyên] ...
    
    def predict_hormone_changes(self, attention_types, status, current_hormone_levels, delta_vector=None):
        # 1. Map status
        status_mapping = {
            "high attend": "high_attend",
            "moderate attend": "moderate_attend",
            "neutral": "neutral",
            "unattend": "unattend",
            "strong unattend": "strong_unattend"
        }
        status_key = status_mapping.get(status, "neutral")

        # 2. Tính hormone đặc thù từ attention
        specific_hormones = {}
        for attention in attention_types:
            if attention in self.ATTENTION_TYPE_HORMONES:
                for hormone, change in self.ATTENTION_TYPE_HORMONES[attention].items():
                    specific_hormones[hormone] = specific_hormones.get(hormone, 0) + change

        # 3. Tính hormone toàn cục từ status
        global_effects = self.ATTENTION_STATUS_HORMONES.get(status_key, {})
        global_hormones = {}
        for effect, value in global_effects.items():
            if '_boost' in effect or '_reduction' in effect or '_penalty' in effect:
                hormone = effect.replace('_boost', '').replace('_reduction', '').replace('_penalty', '')
                sign = 1 if '_boost' in effect else -1
                global_hormones[hormone] = global_hormones.get(hormone, 0) + sign * value

        # 4. Kết hợp hormone từ attention và status
        combined_hormones = {}
        for hormone, value in specific_hormones.items():
            combined = value
            if hormone in global_hormones:
                combined *= self.STATUS_IMPACT_FACTOR
            combined_hormones[hormone] = combined

        for hormone, value in global_hormones.items():
            combined_hormones[hormone] = combined_hormones.get(hormone, 0) + value

        # 5. Trường hợp đặc biệt
        if 'cognitive_conflict' in attention_types:
            combined_hormones['glutamate'] = combined_hormones.get('glutamate', 0) + 0.4
            combined_hormones['GABA'] = combined_hormones.get('GABA', 0) + 0.3

        if 'comforted_by_caregiver' in attention_types:
            combined_hormones['cortisol'] = combined_hormones.get('cortisol', 0) * 0.4 if 'cortisol' in combined_hormones else -0.3

        if self.medical_conditions.get('is_sleeping', False):
            combined_hormones['cortisol'] = combined_hormones.get('cortisol', 0) - 0.5

        # 6. Điều chỉnh theo mức chú ý và tình trạng hiện tại
        status_factors = {
            "high attend": 1.0,
            "moderate attend": 0.8,
            "neutral": 0.5,
            "unattend": 0.1,
            "strong unattend": 0.05
        }
        status_factor = status_factors.get(status, 0.5)

        has_strong_negative_stimulus = any(attn in attention_types for attn in [
            'tactile_discomfort', 'startled_by_loud_noise', 'bitter_taste_rejection',
            'heat_discomfort', 'cold_stress'
        ])

        final_changes = {}
        for hormone, change in combined_hormones.items():
            current_level = current_hormone_levels.get(hormone, 0.0)

            # Diminishing Returns
            diminishing_factor = max(0.1, 1 - (current_level / 2.0)**2)
            adjusted_change = change * diminishing_factor

            # Không áp dụng status factor cho hormone phản ứng stress
            if hormone not in ['cortisol', 'adrenaline', 'norepinephrine']:
                adjusted_change *= status_factor

            # Ức chế hormone tích cực khi có kích thích tiêu cực
            if has_strong_negative_stimulus and hormone in ['oxytocin', 'serotonin', 'dopamine', 'endorphins']:
                self.logger.debug(f"Suppressing positive hormone {hormone} due to negative stimulus.")
                adjusted_change *= 0.1

            final_changes[hormone] = adjusted_change
        if delta_vector is not None:
            # Ánh xạ delta_vector (8 chiều) sang các hormone
            # Ví dụ: mỗi chiều của delta_vector tăng/giảm một hormone tương ứng
            delta_to_hormone_map = {
                0: 'dopamine',
                1: 'GABA',
                2: 'serotonin',
                3: 'cortisol',
                4: 'oxytocin',
                5: 'glutamate',
                6: 'adrenaline',
                7: 'endorphins',
            }
            
            for dim, value in enumerate(delta_vector):
                hormone_name = delta_to_hormone_map.get(dim)
                if hormone_name:
                    # Giả sử mỗi đơn vị delta_vector tương ứng một thay đổi 0.5 lần giá trị
                    change = value * 0.5
                    if hormone_name in combined_hormones:
                        combined_hormones[hormone_name] += change
                    else:
                        combined_hormones[hormone_name] = change
        

        return final_changes
    def _update_emotional_state_from_delta(self, delta_vector):
        """
        Cập nhật self.memory_system.current_emotional_state từ delta_vector.
        Đây là cầu nối giữa "cảm giác" (delta) và "nhận thức về cảm giác" (emotional state).
        """
        # Bản đồ từ chỉ số của delta_vector sang tên cảm xúc
        delta_to_emotion_map = {
            0: 'vui_ve',       # dopamine
            1: 'uc_che',       # GABA
            2: 'hai_long',     # serotonin
            3: 'cang_thang',   # cortisol
            4: 'an_toan',      # oxytocin
            5: 'kich_thich',   # glutamate
            6: 'hoi_hop',      # adrenaline
            7: 'thoai_mai'     # endorphins
        }

        # Phân rã trạng thái cũ
        for emotion in list(self.memory_system.current_emotional_state.keys()):
            self.memory_system.current_emotional_state[emotion] *= 0.90  # Phân rã nhanh hơn
            if self.memory_system.current_emotional_state[emotion] < 0.01:
                del self.memory_system.current_emotional_state[emotion]

        # Cập nhật trạng thái mới từ delta_vector
        for i, value in enumerate(delta_vector):
            if abs(value) > 0.1:  # Chỉ cập nhật nếu tín hiệu đủ mạnh
                emotion_name = delta_to_emotion_map.get(i)
                if emotion_name:
                    # Giá trị dương làm tăng cảm xúc tương ứng
                    if value > 0:
                        current_val = self.memory_system.current_emotional_state.get(emotion_name, 0)
                        self.memory_system.current_emotional_state[emotion_name] = min(1.0, current_val + abs(value) * 0.5)

        self.logger.debug(f"Updated emotional state from delta: {self.memory_system.current_emotional_state}")

    def start_audio_processing(self):
        """Bắt đầu xử lý âm thanh"""
        self.audio_processor.start()
    
    def stop_audio_processing(self):
        """Dừng xử lý âm thanh"""
        self.audio_processor.stop()
    def _form_consciousness_input(self, hormones, sensory, memory_embedding):
        """Tổng hợp vector đầu vào 128D cho tầng thức"""
        # 1. Trạng thái nội tại (hormone + cảm xúc) - 15D
        internal_state = [
            hormones.get('cortisol', 0),
            hormones.get('dopamine', 0),
            hormones.get('oxytocin', 0),
            hormones.get('serotonin', 0),
            hormones.get('adrenaline', 0),
            hormones.get('endorphins', 0),
            hormones.get('GABA', 0),
            self.current_emotional_state.get('vui_ve', 0),
            self.current_emotional_state.get('cang_thang', 0),
            self.current_emotional_state.get('so_hai', 0),
            self.current_emotional_state.get('an_toan', 0),
            self.emotion_integrator.personality_vector.get('HY', 0),  # Hỷ
            self.emotion_integrator.personality_vector.get('no', 0),  # Nộ
            self.emotion_integrator.personality_vector.get('ai', 0),  # Ái
            self.emotion_integrator.personality_vector.get('O', 0),   # Ố
        ]
        
        # 2. Môi trường (đặc trưng cảm giác nổi bật) - 10D
        env_features = [
            sensory['camera'].get('motion', 0),
            sensory['camera'].get('brightness', 0),
            sensory['camera'].get('face_detected', 0),
            sensory['camera'].get('object_size', 0),
            sensory['mic'].get('volume_dB', 0) / 100,  # Chuẩn hóa
            sensory['mic'].get('is_voice', 0),
            1 if 'comforted_by_caregiver' in sensory.get('attention_types', []) else 0,
            sensory.get('temperature_C', 37) / 40,  # Chuẩn hóa
            len(sensory.get('force', [])),  # Số điểm tiếp xúc
            sensory.get('cognitive', {}).get('conflict_level', 0)
        ]
        
        # 3. Bối cảnh bộ nhớ (embedding) - 64D (giả sử memory_embedding là vector 64D)
        memory_vector = memory_embedding.flatten()[:64]  # Đảm bảo đúng kích thước
        
        # Tổng hợp thành vector 89D (15 + 10 + 64)
        combined = np.concatenate([internal_state, env_features, memory_vector])
        
        # Padding nếu cần thiết để đạt 128D
        if len(combined) < 128:
            padding = np.zeros(128 - len(combined))
            combined = np.concatenate([combined, padding])
        
        logger.debug(f"Tạo vector thức: nội tại={len(internal_state)}D, môi trường={len(env_features)}D, bộ nhớ={len(memory_vector)}D")
        return combined

    def _check_consciousness_reset(self, attention_types):
        """Kiểm tra sự kiện đòi hỏi reset ý thức"""
        reset_events = {
            'startled_by_loud_noise',    # Giật mình bởi tiếng ồn lớn
            'strong_unattend',           # Không chú ý mạnh
            'sleep_state_change',        # Thay đổi trạng thái ngủ
            'basic_needs_required',      # Nhu cầu cơ bản (đói, đau)
            'emergency_attention',       # Tình huống khẩn cấp
            'object_reappearance_surprise' # Vật thể tái xuất hiện bất ngờ
        }
        
        # Kiểm tra nếu có bất kỳ sự kiện reset nào
        needs_reset = bool(reset_events & set(attention_types))
        
        if needs_reset:
            logger.warning(f"YÊU CẦU RESET Ý THỨC do sự kiện: {reset_events & set(attention_types)}")
        
        return needs_reset

    def _select_action(self, active_intent):
        """Chọn hành động từ vector ý động"""
        # Chuyển tensor thành mảng numpy
        intent_probs = F.softmax(active_intent, dim=-1).detach().numpy().flatten()
        
        # Định nghĩa các hành động tương ứng
        actions = [
            'EXPLORE_OBJECT',     # Khám phá vật thể
            'AVOID_STIMULUS',     # Tránh kích thích
            'SEEK_CAREGIVER',     # Tìm người chăm sóc
            'VOCALIZE',           # Phát âm thanh
            'REST',               # Nghỉ ngơi
            'REPEAT_ACTION',      # Lặp lại hành động trước
            'FOCUS_ATTENTION',    # Tập trung chú ý
            'IGNORE_STIMULUS'     # Phớt lờ kích thích
        ]
        
        # Chọn hành động ngẫu nhiên theo xác suất
        chosen_idx = np.random.choice(len(actions), p=intent_probs)
        chosen_action = actions[chosen_idx]
        
        logger.info(f"Chọn hành động: {chosen_action} (Xác suất: {intent_probs[chosen_idx]:.2f})")
        return chosen_action

    def _evaluate_action_success(self, action_taken, scores):
        """Đánh giá thành công của hành động dựa trên điểm cảm biến"""
        # Logic đơn giản: hành động thành công nếu điểm attention tăng
        attention_score = scores.get('attention_score', 0)
        prev_score = getattr(self, '_prev_attention_score', 0)
        self._prev_attention_score = attention_score
        
        # Đánh giá theo loại hành động
        if action_taken in ['EXPLORE_OBJECT', 'FOCUS_ATTENTION']:
            success = attention_score > prev_score
        elif action_taken in ['AVOID_STIMULUS', 'IGNORE_STIMULUS']:
            success = attention_score < prev_score
        else:
            success = True  # Mặc định thành công cho các hành động khác
        
        logger.debug(f"Đánh giá hành động {action_taken}: {'Thành công' if success else 'Thất bại'} "
                    f"(Điểm: {attention_score:.2f} vs Trước: {prev_score:.2f})")
        return success
    def step(self, sensory_input):
        """Thực hiện một bước hoàn chỉnh với luồng logic đã được sửa lỗi."""
        self.logger.debug("--- New Step ---")
        # --- 0. Thiết lập thời gian ---
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        self.logger.debug(f"Delta time: {delta_time:.4f}s")
        # 0. Kiểm tra ấn tượng mạnh
        if self._is_strong_impression(sensory_input):
            self._handle_strong_impression()
        # --- 1. Tiền xử lý đầu vào ---

        if self.audio_processor:
            audio_features = self.audio_processor.get_audio_features()
            if audio_features:
                sensory_input['mic'] = audio_features
                contextual_features = self.audio_processor.get_contextual_features()
                if contextual_features:
                    # Gộp vào mic để dễ truy cập
                    sensory_input['mic']['audio_context'] = contextual_features

        if 'raw_frame' in sensory_input and self.vision_processor:
            vision_features = self.vision_processor.process_frame(sensory_input['raw_frame'])
            sensory_input['camera'] = vision_features
        self.hormone_system.apply_feedback(sensory_input)    
        sensory_input = self._apply_gaba_inhibition(sensory_input)

        # Áp dụng phản hồi từ hormone lên đầu vào cảm biến (nếu có)
        sensory_input.setdefault('attention_types', set())
        

        # --- 2. Tính toán chú ý ---
        score, status, scores, types = self.calculate_attention(sensory_input)
        self.logger.info(f"Attention calculated: Score={score:.2f}, Status='{status}', Types={types}")
                #emotion_delta = self.delta_module.compute_delta(types)  
                #self.logger.debug(f"Individual sensor scores: {scores}")
        emotion_delta = self.delta_generator.generate_delta(types)      
        # === THAY ĐỔI: SỬ DỤNG DELTA GENERATOR ===
        # Tạo delta vector từ các attention types
        delta_vector = self.delta_generator.generate_delta(types)
        self.logger.debug(f"Generated delta vector: {delta_vector}")  
         #CẬP NHẬT TRẠNG THÁI CẢM XÚC (LOGIC MỚI) === 
        self._update_emotional_state_from_delta(delta_vector)     
        # Xử lý xung đột nhận thức (nếu có)
        conflict_level = sensory_input.get('cognitive', {}).get('conflict_level', 0)
        if conflict_level > 0.7:
            types.add('cognitive_conflict')
            self.logger.info("Cognitive conflict detected.")

        # Áp dụng phản hồi GABA lên điểm chú ý
        types, scores = self._apply_gaba_feedback(types, scores,) 
        # --- 3. Dự đoán thay đổi Hormone từ Chú ý ---
        attention_hormone_changes = self.predict_hormone_changes(types, status,self.hormone_system.get_summary(), delta_vector)
        self.logger.debug(f"Predicted hormone changes from attention: {attention_hormone_changes}")
        # --- 4. Điều hòa và tích hợp cảm xúc (Ampha) ---
        # Cập nhật trạng thái cảm xúc nội tại của mô hình dựa trên hormone từ bước TRƯỚC
        
        self.harmony_bridge.attention_model.current_emotional_state = self.memory_system.current_emotional_state
        # Tạo/cập nhật ampha dựa trên trạng thái cảm xúc hiện tại
        self.current_ampha = self.harmony_bridge.step() 
        if self.current_ampha:
            self.logger.info(f"Dominant Ampha: {self.current_ampha['key']} (Intensity: {self.current_ampha['intensity']:.2f})")
            # === CẬP NHẬT TÍNH CÁCH ===
            # Cập nhật vector tính cách dựa trên ampha trội và cường độ của nó
            self.emotion_integrator.update_personality(
                self.current_ampha['key'],
                intensity_delta=0.02 * self.current_ampha['intensity'] # Cường độ càng cao, ảnh hưởng càng lớn
            )        
        self.logger.info(f"Current Ampha generated: {self.current_ampha['key'] if self.current_ampha else 'None'}")
        # Lấy thay đổi hormone từ ampha
        ampha_hormone_changes = self.harmony_bridge.apply_ampha_to_hormones()
        self.logger.debug(f"Predicted hormone changes from ampha: {ampha_hormone_changes}")

        # --- 5. Cập nhật hệ thống Hormone (MỘT LẦN DUY NHẤT) ---
        # Kết hợp các thay đổi từ chú ý và ampha
        combined_hormone_changes = attention_hormone_changes.copy()
        for hormone, change in ampha_hormone_changes.items():
            combined_hormone_changes[hormone] = combined_hormone_changes.get(hormone, 0) + change
        self.logger.debug(f"Combined hormone changes: {combined_hormone_changes}")
    

        # Cập nhật hormone với tổng các thay đổi
        self.hormone_system.update_levels(combined_hormone_changes, delta_time)
        hormone_summary = self.hormone_system.get_summary() # Lấy trạng thái hormone MỚI NHẤT
        self.logger.info(f"Updated hormone levels: { {k: f'{v:.2f}' for k, v in hormone_summary.items()} }")
        # --- 6. Xử lý bộ nhớ và trải nghiệm ---
        experience_embedding = self.memory_system.encode_experience(
            sensory_input,
            (score, status, types),
            hormone_summary,
            self.current_ampha  # Truyền ampha hiện tại vào bộ nhớ
        )
        personality_profile = self.emotion_integrator.get_personality_profile()
        self.memory_system.store_experience(
            experience_embedding,
            hormone_summary,
            types,
            score,
            self.current_ampha,
            personality_profile,
             delta_vector  # <--- THÊM THAM SỐ NÀY VÀO
        )
        self.memory_system.update_weights_based_on_hormones(hormone_summary, delta_time)

        if self.medical_conditions['is_sleeping']:
            self.memory_system.consolidate_memory()

        related_memories, recall_triggered = self.memory_system.retrieve_related_experiences(
            experience_embedding, 
            personality_profile , delta_vector # <--- THÊM THAM SỐ NÀY VÀO
        )

        # --- 7. Cập nhật các hệ thống phụ thuộc ---
        # Học liên kết cảm xúc
        self.memory_system.update_emotion_associations(experience_embedding, hormone_summary, self.current_ampha)

        # Cập nhật trạng thái sinh lý
        physiological_state = self.physio_center.update(hormone_summary, ampha=self.current_ampha)
        self.logger.info(f"Updated physiological state: {physiological_state}")
        # === XỬ LÝ TẦNG THỨC ===
        # Tạo vector đầu vào cho thức (nội tại + môi trường + bộ nhớ)
        consciousness_input = self._form_consciousness_input(
            hormone_summary, 
            sensory_input, 
            experience_embedding
        )
        # Đảm bảo định dạng đúng cho LSTM
        consciousness_tensor = torch.tensor(consciousness_input, dtype=torch.float32).view(1, 1, -1)
        # Kiểm tra sự kiện reset (giật mình, thay đổi trạng thái lớn)
        reset_flag = self._check_consciousness_reset(types)
        
        # Xử lý qua module thức
        active_intent, mental_disposition = self.consciousness(
            consciousness_tensor,
            reset_flag
        )
        
        # Ghi nhận ý định và ý tâm
        logger.info("Ý TÂM (Mental Disposition): %s", mental_disposition.detach().numpy())
        logger.info("Ý ĐỘNG (Active Intent): %s", active_intent.detach().numpy())
        
        # Chọn hành động dựa trên ý định
        chosen_action = self._select_action(active_intent)
        
        # Đánh giá phần thưởng
        action_success = self._evaluate_action_success(chosen_action, scores)
        reward = self.reward_evaluator.compute_reward(
            hormone_summary,
            chosen_action,
            action_success
        )        
        # 5. Cập nhật neo cảm xúc và sở thích
        if self.current_ampha and self.current_ampha['intensity'] > 0.6:
            # Tạo neo cảm xúc nếu chưa có
            if not self.preference_memory.reinforce_anchor(
                self.current_ampha['key'], 
                self._extract_sensory_pattern(sensory_input)
            ):
                self.preference_memory.create_emotion_anchor(
                    self.current_ampha['key'], 
                    self._extract_sensory_pattern(sensory_input)
                )
                
            # Cập nhật sở thích với phần thưởng dương
            self.preference_memory.update_preference(
                self._extract_sensory_pattern(sensory_input),
                self.current_ampha['intensity']  # Phần thưởng tỷ lệ với cường độ ampha
            )
            self.logger.debug(f"Emotion anchor updated for ampha: {self.current_ampha['key']}")
        
        # 6. Tìm kiếm kích thích phù hợp với ampha mong muốn
        if self.desired_ampha:
            preferred_stimuli = self.preference_memory.get_preferred_stimuli(self.desired_ampha)
            self._guide_attention(sensory_input, preferred_stimuli)        
        # --- 7. Học hỏi và phát triển cá tính ---
        self._update_personality()
        self.logger.debug("--- Step End ---")
        # --- 8. Trả về kết quả ---
        return score, status, scores, types, combined_hormone_changes, hormone_summary, physiological_state
    def _update_personality(self):
        """Cập nhật đặc điểm cá tính dựa trên lịch sử ampha"""
        if not self.current_ampha:
            return
            
        # Tăng cường ampha thường xuất hiện
        for ampha_key in list(self.amphas.keys()):
            if ampha_key == self.current_ampha['key']:
                self.personality_traits[ampha_key] = min(
                    1.0, 
                    self.personality_traits.get(ampha_key, 0) + 0.05 * self.current_ampha['intensity']
                )
        
        # Giảm các ampha ít xuất hiện
        for trait in self.personality_traits:
            if trait != self.current_ampha['key']:
                self.personality_traits[trait] *= 0.98    
    def _is_strong_impression(self, sensory_input):
        """Xác định xem có kích thích mạnh không"""
        # Âm thanh lớn
        if sensory_input.get('mic', {}).get('volume_dB', 0) > 80:
            return True
            
        # Chuyển động nhanh
        if sensory_input.get('camera', {}).get('motion', 0) > 0.8:
            return True
            
        # Ánh sáng chói
        if sensory_input.get('camera', {}).get('brightness', 0) > 0.9:
            return True
            
        return False
        
    def _handle_strong_impression(self):
        """Xử lý kích thích mạnh - phá vỡ trạng thái hiện tại"""
        if self.current_ampha and self.current_ampha['intensity'] > 0.7:
            # Giảm cường độ ampha hiện tại
            self.hormone_system.apply_immediate_change({
                self.current_ampha['key']: -0.4
            })
            
        # Đặt lại ampha mong muốn
        self.desired_ampha = None
        
        # Tăng cảnh giác
        self.hormone_system.apply_immediate_change({
            'adrenaline': 0.3,
            'cortisol': 0.2
        })    
    def set_desired_ampha(self, ampha_key):
        """Thiết lập ampha mong muốn (từ hệ thống điều khiển cao cấp)"""
        self.desired_ampha = ampha_key
        
    def _extract_sensory_pattern(self, sensory_input):
        """Trích xuất mẫu cảm giác đặc trưng"""
        return {
            'color': sensory_input.get('camera', {}).get('hue_R', 0),
            'sound': sensory_input.get('mic', {}).get('sound_type', 'unknown'),
            'motion': sensory_input.get('camera', {}).get('motion', 0)
        }    
    def _guide_attention(self, sensory_input, preferred_stimuli):
        """Định hướng sự chú ý dựa trên sở thích"""
        # Triển khai thực tế sẽ so sánh sensory_input với preferred_stimuli
        # và điều chỉnh trọng số cảm biến tương ứng
        pass    
    def _calculate_vision_score(self, camera):
        """Tính điểm thị giác sử dụng SensorProcessor với logging chi tiết"""
        if not camera: return 0.0, set()

        thresholds = self._get_safety_threshold('vision')
        attention_types = set() 
        score = 0.0 
        debug_info = [] # Bảng tin debug
    # 1. Phân tích chuyển động có chủ đích
        for obj in camera.get('tracked_objects', []):
            trajectory = obj.get('trajectory', [])
            if len(trajectory) > 2:
                # Tính toán độ lệch quỹ đạo
                deviations = []
                for i in range(1, len(trajectory)-1):
                    prev_point = trajectory[i-1]
                    curr_point = trajectory[i]
                    next_point = trajectory[i+1]
                    
                    # Tính vector hướng dự kiến và thực tế
                    expected_vector = (next_point[0]-prev_point[0], next_point[1]-prev_point[1])
                    actual_vector = (curr_point[0]-prev_point[0], curr_point[1]-prev_point[1])
                    
                    # Tính góc lệch
                    dot_product = expected_vector[0]*actual_vector[0] + expected_vector[1]*actual_vector[1]
                    mag_expected = math.sqrt(expected_vector[0]**2 + expected_vector[1]**2)
                    mag_actual = math.sqrt(actual_vector[0]**2 + actual_vector[1]**2)
                    
                    if mag_expected > 0 and mag_actual > 0:
                        cos_angle = dot_product / (mag_expected * mag_actual)
                        angle_dev = math.acos(min(1, max(-1, cos_angle)))
                        deviations.append(angle_dev)
                
                # Phân loại chuyển động
                if deviations:
                    avg_deviation = sum(deviations) / len(deviations)
                    if avg_deviation < 0.2:  # Radian (~11.5 độ)
                        attention_types.add('predictable_motion_focus')
                        debug_info.append(f"+PredictableMotion({obj.get('id',-1)}):{bonus:.2f}")
    
                        score += 0.3 * obj.get('stability', 0)
                    else:
                        attention_types.add('erratic_motion_alert')
                        debug_info.append(f"+ErraticMotion({obj.get('id',-1)}):0.10")

                        score += 0.1  # Cảnh báo quan trọng nhưng không khuyến khích
        
        # 2. Hiểu sự phức tạp của vật thể
        for proto in camera.get('proto_objects', []):
            complexity = proto.get('complexity', 0)
            components = proto.get('components', [])
            
            if complexity > 1.5 and proto.get('stability', 0) > 0.8:
                # Kiểm tra xem có hình dạng khuôn mặt không
                face_like = any('face' in comp for comp in components)
                attention_types.add('complex_familiar_pattern_recognition')
                debug_info.append(f"+ComplexFamiliar:{bonus:.2f}")
                score += 0.4 * complexity * (1.2 if face_like else 1.0)
            elif complexity > 1.5:
                attention_types.add('novel_complex_stimulus_curiosity')
                debug_info.append(f"+NovelComplex:{bonus:.2f}")
                
                score += 0.2 * complexity
        
        # 3. Suy luận từ quan hệ không gian
        spatial_relations = camera.get('spatial_relations', {})
        geometric_features = camera.get('geometric_features', [])
        
        for rel_id, relations in spatial_relations.items():
            for target_id, rel_data in relations.items():
                relation_type = rel_data.get('relation', '')
                
                # Lấy thông tin hình dạng
                rel_shape = geometric_features[rel_id].get('shape_type', '') if rel_id < len(geometric_features) else ''
                target_shape = geometric_features[target_id].get('shape_type', '') if target_id < len(geometric_features) else ''
                
                if relation_type == 'contains':
                    if target_shape == 'circle' and rel_shape in ['square', 'rectangle']:
                        attention_types.add('object_containment_logic')
                        score += 0.5
                elif relation_type == 'adjacent':
                    if rel_shape == target_shape and rel_shape in ['square', 'circle']:
                        attention_types.add('pattern_repetition_interest')
                        score += 0.3 * camera.get('motion', 0)        
        # === Điều chỉnh theo độ tuổi ===
        age_group = self._get_age_group()
        
        # 0-3 tháng
        if age_group <= 1:
            # Tăng cường khuôn mặt và hình đơn giản
            if camera.get('face_detected', False):
                score += 0.4
                attention_types.add('infant_face_preference')
            
            # Giảm tác động của yếu tố phức tạp
            if any(t in attention_types for t in ['complex_familiar_pattern_recognition', 'novel_complex_stimulus_curiosity']):
                original_score = score
                score *= 0.7
                debug_info.append(f"*AgeDebuff_Complex:{(score-original_score):.2f}")
        
        # 4-8 tháng (Object Permanence)
        elif age_group == 2:
            # Tăng cường vật thể được theo dõi ổn định
            for obj in camera.get('tracked_objects', []):
                if obj.get('stability', 0) > 0.7:
                    score += 0.3
                    attention_types.add('object_permanence_learning')
            
            # Tăng điểm khi vật thể tái xuất hiện
            if any('reappeared' in obj for obj in camera.get('tracked_objects', [])):
                score += 0.4
                attention_types.add('object_reappearance_surprise')
        
        # 9-18 tháng (Không gian logic)
        elif age_group >= 3:
            # Tăng cường quan hệ không gian
            spatial_multiplier = 1.0
            if 'object_containment_logic' in attention_types:
                spatial_multiplier *= 1.5
            if 'pattern_repetition_interest' in attention_types:
                spatial_multiplier *= 1.3
            
            score *= spatial_multiplier
            
            # Khuyến khích thử nghiệm không gian
            if camera.get('motion', 0) > 0.6 and any('container' in obj for obj in camera.get('proto_objects', [])):
                score += 0.4
                attention_types.add('spatial_experimentation')        
        # === Phản hồi từ trạng thái nội tại ===
        hormone_levels = self.hormone_system.get_summary()
        
        # 1. Phản hồi từ Hormone
        cortisol = hormone_levels.get('cortisol', 0)
        dopamine = hormone_levels.get('dopamine', 0)
        oxytocin = hormone_levels.get('oxytocin', 0)
        
        # Cortisol cao -> cảnh giác với chuyển động
        if cortisol > 0.7:
            if 'erratic_motion_alert' in attention_types:
                score += 0.5
                debug_info.append(f"+Cortisol_ErraticBoost:0.50")

            # Giảm hứng thú với vật thể phức tạp
            if 'novel_complex_stimulus_curiosity' in attention_types:
                score -= 0.3
                debug_info.append(f"-Cortisol_NoveltyDebuff:0.30")

        
        # Dopamine cao -> tăng hứng thú khám phá
        if dopamine > 0.6:
            novelty_bonus = 0.4 if 'novel_complex_stimulus_curiosity' in attention_types else 0
            debug_info.append(f"+Dopamine_NoveltyBoost:0.40")
            motion_bonus = 0.3 * camera.get('motion', 0)
            score += novelty_bonus + motion_bonus
        
        # Oxytocin cao -> tập trung vào kết nối
        if oxytocin > 0.7:
            if camera.get('face_detected', False):
                # Tăng thêm nếu là khuôn mặt quen thuộc
                familiarity_bonus = 0.6 if any(obj.get('familiar', False) 
                                            for obj in camera.get('tracked_objects', [])) else 0.3
                score += familiarity_bonus
        
        # 2. Phản hồi từ Sở thích (Preference Memory)
        if self.preference_memory and self.current_ampha:
            # Lấy các kích thích ưa thích liên quan đến ampha hiện tại
            preferred_stimuli = self.preference_memory.get_preferred_stimuli(self.current_ampha['key'])
            
            # Tạo profile thị giác hiện tại
            current_profile = {
                'dominant_shape': self._get_dominant_shape(camera),
                'color_profile': (camera.get('hue_R', 0), camera.get('hue_G', 0), camera.get('hue_B', 0)),
                'motion_level': camera.get('motion', 0)
            }
            
            # Kiểm tra khớp với sở thích
            for stimulus in preferred_stimuli:
                if self._stimulus_similarity(current_profile, stimulus) > 0.6:
                    score *= 1.8  # Tăng mạnh điểm
                    attention_types.add('preference_match')
                    debug_info.append(f"*PreferenceBoost:{(score-original_score):.2f}")
                    break           
        # Log bảng tin debug trước khi trả về
        logger.debug(f"Vision score breakdown: [{' | '.join(debug_info)}]")                         
        final_score = max(-0.5, min(1.0, score))
        return final_score, attention_types
    def _get_dominant_shape(self, camera):
        """Xác định hình dạng chiếm ưu thế trong khung hình"""
        shape_counter = {}
        for feature in camera.get('geometric_features', []):
            shape = feature.get('shape_type', 'unknown')
            shape_counter[shape] = shape_counter.get(shape, 0) + 1
        
        if shape_counter:
            return max(shape_counter, key=shape_counter.get)
        return 'none'

    def _stimulus_similarity(self, profile1, profile2):
        """Tính độ tương đồng giữa hai profile kích thích"""
        # So sánh hình dạng
        shape_sim = 1.0 if profile1['dominant_shape'] == profile2['dominant_shape'] else 0.3
        
        # So sánh màu sắc (khoảng cách Euclidean)
        color_diff = sum((a - b)**2 for a, b in zip(profile1['color_profile'], profile2['color_profile']))**0.5
        color_sim = max(0, 1 - color_diff / math.sqrt(3))  # Chuẩn hóa về [0,1]
        
        # So sánh mức độ chuyển động
        motion_sim = 1 - abs(profile1['motion_level'] - profile2['motion_level'])
        
        # Kết hợp với trọng số
        return 0.5*shape_sim + 0.3*color_sim + 0.2*motion_sim    
    def _calculate_hearing_score(self, mic):
        """Tính điểm thính giác và xác định attention type (cập nhật cho module âm thanh mới)"""
        if not mic: return 0.0, set()
        thresholds = self._get_safety_threshold('hearing')
        max_db = thresholds['max_db'] * (1 + self.age_months/24) 
        score = 0.0
        attention_types = set()
        if mic['volume_dB'] > max_db:
            return -0.5, {'startled_by_loud_noise'}        
        # Kiểm tra ngưỡng an toàn (sử dụng volume_dB từ module mới)
        if mic['volume_dB'] > thresholds['max_db']:
            attention_types.add('startled_by_loud_noise')
            return -0.5, attention_types
        
        # Sử dụng dominant_frequency_Hz thay cho frequency_Hz
        dominant_freq = mic.get('dominant_frequency_Hz', 0)
        if not (thresholds['min_freq'] <= dominant_freq <= thresholds['max_freq']):
            return -0.3, set()
        
        # Xử lý thông tin giọng nói từ module mới
        if mic.get('is_voice', False):
            if mic.get('voice_probability', 0) > 0.7:
                score += 0.6
                attention_types.add('seeking_mother_voice')
            else:
                score += 0.4
        elif mic.get('is_rhythmic', False):
            score += 0.5
            attention_types.add('rhythmic_sound_engagement')
        
        # Phân loại âm thanh dựa trên kết quả từ module mới
        sound_type = mic.get('sound_type', 'unknown')
        if sound_type == 'lullaby':
            score += 0.4
            attention_types.add('soothing_sound_focus')
        elif sound_type == 'high_pitch':
            score -= 0.2
            attention_types.add('high_pitch_discomfort')
        
        # Âm lượng tối ưu (50-65 dB)
        if 50 <= mic['volume_dB'] <= 65:
            score += 0.4
        else:
            score += 0.2
        
        # Xem xét xu hướng âm lượng từ ngữ cảnh
        if 'audio_context' in mic:
            context = mic['audio_context']
            if context.get('volume_trend', 0) == 1:  # Âm lượng đang tăng
                attention_types.add('increasing_volume_interest')
                score += 0.1
            elif context.get('is_continuous_speech', False):
                attention_types.add('continuous_speech_engagement')
                score += 0.2
        
        return min(1.0, max(-0.5, score)), attention_types
    
    def _calculate_smell_score(self, smell_compounds):
        """Tính điểm khứu giác và xác định attention type"""
        thresholds = self._get_safety_threshold('smell')
        score = 0.0
        attention_types = set()
        
        for compound in smell_compounds:
            conc = compound['concentration_ppm']
            
            # Kiểm tra ngưỡng an toàn
            if conc > thresholds['max_concentration'] * 3:
                attention_types.add('odor_aversion')
                return -0.6, attention_types
            
            # Hợp chất tích cực
            if compound['compound'] in self.SMELL_COMPOUNDS['positive']:
                if conc <= thresholds['max_concentration']:
                    score += 0.4 * min(1.0, conc/thresholds['max_concentration'])
                    if 'nonanal' in compound['compound'] or 'heptanal' in compound['compound']:
                        attention_types.add('seeking_maternal_scent')
                    else:
                        attention_types.add('familiar_food_attraction')
            
            # Hợp chất tiêu cực
            elif compound['compound'] in self.SMELL_COMPOUNDS['negative']:
                if conc > thresholds['max_concentration']:
                    score -= 0.3 * min(1.0, conc/thresholds['max_concentration'])
                    attention_types.add('odor_aversion')
            # Mùi lạ nhưng không nguy hiểm
            else:
                attention_types.add('novel_odor_exploration')
                score += 0.2 * min(1.0, conc/thresholds['max_concentration'])
        
        return max(-0.5, min(1.0, score)), attention_types
    
    def _calculate_taste_score(self, taste_compounds):
        """Tính điểm vị giác và xác định attention type"""
        thresholds = self._get_safety_threshold('taste')
        score = 0.0
        attention_types = set()
        
        for compound in taste_compounds:
            conc = compound['concentration_mg_dl']
            
            # Vị ngọt (ưa thích)
            if compound['compound'] in self.TASTE_COMPOUNDS['positive']:
                score += min(1.0, conc/100.0) * 0.6
                attention_types.add('sweet_taste_seeking')
            
            # Vị đắng (không thích)
            elif compound['compound'] in self.TASTE_COMPOUNDS['negative']:
                if conc > thresholds['max_bitter']:
                    attention_types.add('bitter_taste_rejection')
                    return -0.7, attention_types
                score -= min(1.0, conc/thresholds['max_bitter']) * 0.4
            # Vị chua
            elif 'sour' in self.TASTE_COMPOUNDS and compound['compound'] in self.TASTE_COMPOUNDS['sour']:
                if conc > thresholds.get('min_sour', 5.0):
                    attention_types.add('sour_taste_response')
                    score -= 0.3 * min(1.0, conc/50.0)  # Giảm điểm vì bé thường không thích vị chua
                             
            # Vị lạ
            elif compound.get('is_novel', False):
                attention_types.add('taste_exploration')
                score += 0.2
        
        return max(-0.5, min(1.0, score)), attention_types
    
    def _calculate_touch_score(self, force_points):
        """Tính điểm xúc giác và xác định attention type"""
        score = 0.0
        attention_types = set()
        
        for point in force_points:
            # Áp lực nhẹ (0.5-2N) - dễ chịu
            if 0.5 <= point['force_N'] <= 2.0:
                # Vùng nhạy cảm
                if point['body_part'] in self.SENSITIVE_AREAS:
                    score += 0.4
                    attention_types.add('soft_contact_engagement')
                else:
                    score += 0.2
                    attention_types.add('deep_pressure_seeking' if point['force_N'] > 1.5 else 'soft_contact_engagement')
            
            # Áp lực mạnh - khó chịu
            elif point['force_N'] > 2.0:
                if point['body_part'] in self.SENSITIVE_AREAS:
                    score -= 0.3 if point['body_part'] in self.SENSITIVE_AREAS else 0.2
                    attention_types.add('tactile_discomfort')
                else:
                    score -= 0.3
                    attention_types.add('tactile_discomfort')
            # Cảm giác nhột/cù
            if point.get('is_tickle', False):
                attention_types.add('tickle_reaction')
                score += 0.3                    
        
        final_score = max(-0.5, min(1.0, score/len(force_points))) if force_points else 0.0
        return final_score, set(attention_types)
    
    def _calculate_temp_score(self, temp_C):
        """Tính điểm nhiệt độ và xác định attention type"""
        thresholds = self._get_safety_threshold('temp')
        min_temp = thresholds['min']
        max_temp = thresholds['max']
        attention_types = set()
        
        # Nhiệt độ lý tưởng
        if min_temp <= temp_C <= max_temp:
            attention_types.add('warmth_seeking')
            return 1.0, attention_types
        
        # Quá lạnh
        if temp_C < min_temp:
            deviation = min_temp - temp_C
            max_dev = 5.0
            attention_types.add('cold_stress')
            return max(0.0, 1.0 - (deviation/max_dev)), attention_types
        
        # Quá nóng
        else:
            deviation = temp_C - max_temp
            max_dev = 7.0
            attention_types.add('heat_discomfort')
            return max(0.0, 1.0 - (deviation/max_dev)), attention_types
def get_emotional_terrain(self):
    """Phân tích địa hình cảm xúc dựa trên personality vector"""
    personality = self.emotion_integrator.get_personality_profile()
    
    # Phân tích địa hình
    terrain = {
        'deep_valleys': [],
        'high_plateaus': [],
        'volatile_slopes': []
    }
    
    for emotion, intensity in personality.items():
        if intensity > 0.8:
            terrain['deep_valleys'].append({
                'emotion': emotion,
                'depth': intensity
            })
        elif intensity > 0.6:
            terrain['high_plateaus'].append({
                'emotion': emotion,
                'height': intensity
            })
        elif intensity > 0.4:
            terrain['volatile_slopes'].append({
                'emotion': emotion,
                'steepness': intensity
            })
    
    # Tính toán độ nhạy cảm tổng thể
    volatility = sum(
        max(0, intensity - 0.5) ** 2 
        for intensity in personality.values()
    )
    
    return {
        'personality_profile': personality,
        'emotional_terrain': terrain,
        'emotional_volatility': min(1.0, volatility * 2)
    }        