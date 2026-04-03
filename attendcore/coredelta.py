# attendcore/coredelta.py
import numpy as np
import logging
from collections import defaultdict , deque
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class DeltaGenerator:
    def __init__(self, emotion_dim=8):
        self.emotion_dim = emotion_dim  # Số chiều của vector cảm xúc
        
        # Bản đồ từ attention type sang vector cảm xúc
        self.attention_to_delta_map = {
            # Vision-based
            'bright_color_interest': [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'motion_tracking': [0.6, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
            'overstimulated_by_flicker': [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.7],
            'face_focused': [0.0, 0.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.0],
            'novel_object_interest': [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'pattern_preference': [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            
            # Hearing-based
            'seeking_mother_voice': [0.0, 0.0, 0.0, 0.0, 0.9, 0.7, 0.0, 0.0],
            'soothing_sound_focus': [0.6, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
            'startled_by_loud_noise': [0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.8, 0.0],
            'rhythmic_sound_engagement': [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0],
            'music_engagement': [0.7, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
            
            # Smell-based
            'seeking_maternal_scent': [0.0, 0.0, 0.0, 0.0, 0.8, 0.6, 0.0, 0.0],
            'familiar_food_attraction': [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            'odor_aversion': [0.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.0, 0.0],
            
            # Taste-based
            'sweet_taste_seeking': [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0],
            'bitter_taste_rejection': [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0],
            'taste_exploration': [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'sour_taste_response': [0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
            
            # Touch-based
            'soft_contact_engagement': [0.0, 0.0, 0.0, 0.0, 0.75, 0.65, 0.0, 0.0],
            'deep_pressure_seeking': [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.5],
            'tactile_discomfort': [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.7, 0.0],
            'tickle_reaction': [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            
            # Temperature-based
            'warmth_seeking': [0.0, 0.0, 0.0, -0.6, 0.0, 0.0, 0.7, 0.0],
            'heat_discomfort': [0.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.0, 0.0],
            'cold_stress': [0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.8, 0.0],
            'rapid_temp_change': [0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
            
            # Medical conditions
            'sleeping': [0.0, 0.0, 0.0, -0.8, 0.0, 0.0, 0.9, 0.0],
            'basic_needs_required': [0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0],
            'separation_anxiety': [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.7, 0.0],
            'comforted_by_caregiver': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            'cognitive_conflict': [0.0, 0.3, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0],
        }
        
        # Khởi tạo vector cảm xúc hiện tại
        self.current_delta = np.zeros(self.emotion_dim)
        
        # Lịch sử delta để theo dõi sự thay đổi
        self.delta_history = deque(maxlen=10)
        
        # Mạng neural để học bản đồ delta (tùy chọn nâng cao)
        self.neural_mapper = self._build_neural_mapper()
    
    def _build_neural_mapper(self):
        """Xây dựng mạng neural để học bản đồ attention-to-delta"""
        return nn.Sequential(
            nn.Linear(len(self.attention_to_delta_map), 32),
            nn.ReLU(),
            nn.Linear(32, self.emotion_dim)
        )
    
    def generate_delta(self, attention_types):
        """Tạo vector delta từ tập hợp các attention types"""
        # Reset delta về 0 trước khi tính toán mới
        delta_vector = np.zeros(self.emotion_dim)
        
        # Tổng hợp tác động từ tất cả các attention types
        for att_type in attention_types:
            if att_type in self.attention_to_delta_map:
                delta_vector += np.array(self.attention_to_delta_map[att_type])
        
        # Giới hạn giá trị trong khoảng [-1, 1]
        delta_vector = np.clip(delta_vector, -1.0, 1.0)
        
        # Cập nhật lịch sử
        self.delta_history.append(delta_vector)
        
        # Tính toán delta hiện tại: trung bình có trọng số của lịch sử (gần đây nhất có trọng số cao hơn)
        weights = np.linspace(0.1, 1.0, len(self.delta_history))
        weights /= weights.sum()
        self.current_delta = np.zeros(self.emotion_dim)
        for i, delta in enumerate(self.delta_history):
            self.current_delta += delta * weights[i]
        
        return self.current_delta.copy()
    
    def get_current_delta(self):
        """Lấy vector delta hiện tại"""
        return self.current_delta.copy()