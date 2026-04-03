from collections import defaultdict
import numpy as np
import logging 
logger = logging.getLogger(__name__)
class HormoneDynamics:
    def __init__(self, *,
                 cortisol_decay=0.15,
                 gaba_ie_ratio=0.25,
                 gaba_thresholds=(0.6, 1.6),
                 **kwargs):

        self.levels = {
            'oxytocin': 0.5,
            'cortisol': 0.3,
            'dopamine': 0.4,
            'serotonin': 0.6,
            'adrenaline': 0.2,
            'endorphins': 0.4,
            'melatonin': 0.3,
            'GABA': 1.0,
            'glutamate': 0.8,
            'norepinephrine': 0.3
        }

        # Half-life của từng hormone
        self.half_lives = {
            'oxytocin': 5, 'cortisol': 3, 'dopamine': 4,
            'serotonin': 6, 'adrenaline': 2, 'endorphins': 5,
            'melatonin': 8, 'GABA': 4, 'glutamate': 3,
            'norepinephrine': 3
        }

        # Cơ chế đối kháng hormone
        self.antagonisms = {
            ('oxytocin', 'cortisol'): 0.7,
            ('serotonin', 'cortisol'): 0.6,
            ('dopamine', 'cortisol'): 0.5,
            ('cortisol', 'oxytocin'): 0.8,
            ('cortisol', 'dopamine'): 0.7,
            ('GABA', 'glutamate'): 0.8,
            ('GABA', 'adrenaline'): 0.7,
            ('GABA', 'cortisol'): 0.6
        }

        # Tác động ngược của hormone
        self.hormone_feedback = {
            'oxytocin': {
                'vision_boost': {'target': 'face_focused', 'factor': 1.5}
            },
            'cortisol': {
                'hearing_sensitivity': {
                    'threshold_factor': 0.7,
                    'impact': 2.0
                }
            }
        }

        # Ức chế của GABA
        self.gaba_inhibition = {
            'dopamine': 0.6, 'adrenaline': 0.75,
            'cortisol': 0.5, 'glutamate': 0.85
        }

        # Các thông số GABA
        self.optimal_IE_ratio = gaba_ie_ratio
        self.GABA_thresholds = gaba_thresholds

        # Lịch sử hormone
        self.history = defaultdict(list)
        # Bộ nhớ điều chỉnh GABA
        self._gaba_adjustment_memory = 0.0
        self.time_since_last_update = 0.0

        self.cortisol_decay = cortisol_decay
        
        # THÊM MỚI: Khởi tạo ampha_effects
        self.ampha_effects = {}

    def calculate_IE_balance(self):
        """Tính toán tỉ lệ ức chế/kích thích"""
        inhibitory = self.levels.get('GABA', 0) * 1.2
        excitatory = (
            self.levels.get('glutamate', 0)
            + self.levels.get('dopamine', 0) * 0.7
            + self.levels.get('adrenaline', 0) * 0.8
            + self.levels.get('norepinephrine', 0) * 0.9
        )
        total = inhibitory + excitatory
        return inhibitory / total if total > 0.001 else 0.5
    # === THÊM PHƯƠNG THỨC MỚI VÀO ĐÂY ===
    def apply_immediate_change(self, changes):
        """
        Áp dụng thay đổi hormone ngay lập tức để phản ứng với các sự kiện đột ngột.
        Bỏ qua chu kỳ phân rã và đối kháng thông thường.
        """
        for hormone, change in changes.items():
            # Đảm bảo hormone tồn tại trong danh sách
            if hormone not in self.levels:
                self.levels[hormone] = 0.0
            
            # Cộng trực tiếp sự thay đổi
            self.levels[hormone] += change
            
            # Đảm bảo nồng độ không bị âm
            self.levels[hormone] = max(0.0, self.levels[hormone])
    def apply_gaba_regulation(self):
        """Tự điều chỉnh GABA dựa trên cân bằng I/E"""
        current_ratio = self.calculate_IE_balance()
        gaba = self.levels.get('GABA', 1.0)
        target_ratio = self.optimal_IE_ratio

        # Sai lệch I/E (dương = ức chế nhiều, âm = kích thích nhiều)
        deviation = current_ratio - target_ratio

        # Ngưỡng không điều chỉnh (dead zone)
        if abs(deviation) < 0.05:
           self._gaba_adjustment_memory *= 0.9  # Phân rã bộ nhớ
           return

        # Tính toán điều chỉnh với hệ số tích lũy
        adjustment_factor = 0.4 * (1 + 0.5 * self._gaba_adjustment_memory)
        original_gaba = self.levels.get('GABA', 1.0)

        if deviation > 0:  # Ức chế quá mức → GIẢM GABA
            adjustment = adjustment_factor * deviation
            new_gaba = gaba - adjustment
            self._gaba_adjustment_memory = min(1.0, self._gaba_adjustment_memory + 0.1)
        else:
            adjustment = adjustment_factor * abs(deviation)
            new_gaba = gaba + adjustment
            self._gaba_adjustment_memory = max(-1.0, self._gaba_adjustment_memory - 0.1)

        # Áp dụng với giới hạn an toàn
        self.levels['GABA'] = max(
            self.GABA_thresholds[0],
            min(self.GABA_thresholds[1], new_gaba)
        )
        logger.debug(f"GABA Regulation: Ratio Deviation={deviation:.2f}, Adjustment={adjustment:.2f}, GABA changed from {original_gaba:.2f} to {self.levels['GABA']:.2f}")

    def update_levels(self, new_changes, delta_time=1.0):
        """Cập nhật nồng độ hormone với xử lý ampha và thứ tự rõ ràng"""
        logger.debug(f"--- Updating hormone levels (delta_time: {delta_time:.2f}) ---")
        logger.debug(f"Incoming changes: {new_changes}")
        
        # 0. Tách thay đổi từ attention và ampha
        ampha_input_changes = {k: v for k, v in new_changes.items() if k in self.ampha_effects}
        attention_changes = {k: v for k, v in new_changes.items() if k not in self.ampha_effects}
        for hormone, change in new_changes.items():
            self.levels[hormone] += change
        
        # 1. Áp dụng phân rã TRƯỚC khi cập nhật thay đổi
        for hormone in list(self.levels.keys()):
            if hormone in self.half_lives:
                half_life = self.half_lives[hormone]
                if half_life > 0:
                    decay_factor = 0.5 ** (delta_time / half_life)
                    self.levels[hormone] *= decay_factor
         # 2. Áp dụng thay đổi từ attention và ampha
        for hormone, change in attention_changes.items():
            self.levels[hormone] = self.levels.get(hormone, 0) + change   
        logger.debug(f"Levels after decay: {self.levels}")
        
        # Lưu hiệu ứng ampha
        for hormone, change in ampha_input_changes.items():
            self.ampha_effects[hormone] = self.ampha_effects.get(hormone, 0) + change
            self.levels[hormone] = self.levels.get(hormone, 0) + change
        # 3. Áp dụng giới hạn ngay lập tức
        for hormone in self.levels:
            self.levels[hormone] = max(0.0, min(2.0, self.levels[hormone]))  # Giới hạn tối đa 2.0
        # Đảm bảo tất cả hormone đều tồn tại
        all_hormones = set(attention_changes.keys()) | set(self.ampha_effects.keys())
        for hormone in all_hormones:
            if hormone not in self.levels:
                self.levels[hormone] = 0.0

        # Kết hợp tất cả thay đổi
        combined_changes = {
            **attention_changes,
            **self.ampha_effects
        }
        logger.debug(f"Combined changes to apply: {combined_changes}")
                
        # 3. Áp dụng cơ chế đối kháng
        for (h1, h2), factor in self.antagonisms.items():
            if h1 in self.levels and h2 in self.levels:
                reduction = min(self.levels[h1] * factor, self.levels[h2])
                self.levels[h2] -= reduction
        
        # 4. Áp dụng ức chế GABA
        gaba_level = self.levels.get('GABA', 0)
        for hormone, factor in self.gaba_inhibition.items():
            if hormone in self.levels:
                inhibition = min(gaba_level * factor, self.levels[hormone])
                self.levels[hormone] -= inhibition
        
        # 5. Áp dụng điều chỉnh GABA tự động
        self.apply_gaba_regulation()
        
        # 6. Xử lý cortisol đặc biệt
        if 'cortisol' in self.levels and self.levels['cortisol'] > 0.5:
            self.levels['cortisol'] -= self.cortisol_decay * (self.levels['cortisol'] - 0.5)
        
        # 7. Giới hạn GABA
        if 'GABA' in self.levels:
            self.levels['GABA'] = max(
                self.GABA_thresholds[0],
                min(self.GABA_thresholds[1], self.levels['GABA'])
            )
        # 8. Giới hạn bằng sigmoid#for hormone in self.levels:# self.levels[hormone] = 1 / (1 + np.exp(-self.levels[hormone]))
        # 9. Cập nhật norepinephrine
        # 9. Cập nhật norepinephrine
        if 'cortisol' in self.levels and self.levels['cortisol'] > 0.6:
            self.levels['norepinephrine'] = min(
                1.0, # Giữ giới hạn trên để tránh tăng vô hạn
                self.levels.get('norepinephrine', 0) + 0.2 * self.levels['cortisol']
            )

        
        # 10. Phân rã hiệu ứng ampha
        self.decay_ampha_effects(delta_time)
        
        logger.debug(f"Final hormone levels: {self.get_summary()}")
        return self.get_summary()

    def decay_ampha_effects(self, delta_time, decay_rate=0.9):
        """Phân rã hiệu ứng ampha theo thời gian"""
        for hormone in list(self.ampha_effects.keys()):
            self.ampha_effects[hormone] *= decay_rate ** delta_time
            # Loại bỏ nếu hiệu ứng quá nhỏ
            if abs(self.ampha_effects[hormone]) < 0.001:
                del self.ampha_effects[hormone]

    def apply_feedback(self, sensory_input):
        """Áp dụng tác động ngược của hormone"""
        # Oxytocin feedback
        if self.levels.get('oxytocin', 0) > 0.5:
            feedback = self.hormone_feedback['oxytocin']['vision_boost']
            if 'attention_factors' in sensory_input:
                if feedback['target'] in sensory_input['attention_factors']:
                    sensory_input['attention_factors'][feedback['target']] *= feedback['factor']

        # Cortisol feedback
        cortisol_level = self.levels.get('cortisol', 0)
        if cortisol_level > 0.4:
            feedback = self.hormone_feedback['cortisol']['hearing_sensitivity']
            if 'mic' not in sensory_input:
                sensory_input['mic'] = {}
            sensory_input['mic']['sensitivity_boost'] = feedback['impact']
            sensory_input['mic']['volume_threshold'] = sensory_input['mic'].get('volume_threshold', 60.0) * feedback['threshold_factor']

        # Cognitive conflict
       # attention_types = sensory_input.get('attention_types', [])
        #if 'cognitive_conflict' in attention_types:
         #   self.pending_changes = {
           #     'GABA': 0.3,
            #    'glutamate': 0.4
            #}

    def get_level(self, hormone):
        return self.levels.get(hormone, 0.0)

    def get_summary(self):
        return self.levels.copy()