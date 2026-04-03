import numpy as np
from collections import deque, defaultdict
from sklearn.neighbors import KDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import math # Thêm math
# --- Thiết lập Logging ---
logger = logging.getLogger(__name__)
class EmotionAssociationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class EmotionAssociationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class ExperienceEncoder:
    def __init__(self, 
                 short_term_capacity=10, 
                 long_term_capacity=1000, 
                 embedding_dim=64, 
                 decay_factor=0.95,
                 long_term_thresh=0.7,
                 emotion_recall_short=0.2,
                 emotion_recall_long=0.15,
                 rebuild_interval=50,
                 emotion_net_dim=32,
                 *args, **kwargs):
        
        self.short_term_memory = deque(maxlen=short_term_capacity)
        self.long_term_memory = []
        self.long_term_weights = []
        self.long_term_emotions = []
        # === TÍCH HỢP MỚI: LƯU NGỮ CẢNH TÍNH CÁCH ===
        self.long_term_personality_contexts = []
        self.embedding_dim = embedding_dim
        self.decay_factor = decay_factor
        self.kdtree = None
        self.rebuild_counter = 0
        self.REBUILD_INTERVAL = rebuild_interval
        
        self.base_long_term_thresh = long_term_thresh
        self.long_term_thresh = long_term_thresh
        self.emotion_recall_short = emotion_recall_short
        self.emotion_recall_long = emotion_recall_long
        
        # Kích thước vector
        self.BASE_FEATURE_SIZE = 16 + 5 + 7 # Sensory(16) + Attention(5) + Hormones(7)
        self.AMPHA_VECTOR_SIZE = 8
        autoencoder_input_dim = self.BASE_FEATURE_SIZE + self.AMPHA_VECTOR_SIZE 
        self.encoder = self._build_autoencoder(autoencoder_input_dim, embedding_dim)        
        
        self.EMOTION_MAPPING = {
            'oxytocin': 'an_toan', 'cortisol': 'cang_thang', 'dopamine': 'vui_ve',
            'serotonin': 'hai_long', 'adrenaline': 'hoi_hop', 'endorphins': 'thoai_mai',
            'melatonin': 'buon_ngu', 'GABA': 'uc_che'
        }
        self.EMOTION_MAPPING_SIZE = len(self.EMOTION_MAPPING)
        
        # Cập nhật input_dim cho mạng học cảm xúc
        new_emotion_net_input_dim = self.embedding_dim + self.EMOTION_MAPPING_SIZE + self.AMPHA_VECTOR_SIZE
        self.emotion_net = EmotionAssociationNetwork(
            input_dim=new_emotion_net_input_dim,
            hidden_dim=emotion_net_dim,
            output_dim=self.EMOTION_MAPPING_SIZE
        )
        self.emotion_optimizer = torch.optim.Adam(self.emotion_net.parameters(), lr=0.001)
        
        self.emotion_to_hormone = {
            'an_toan': {'oxytocin': 0.3}, 'cang_thang': {'cortisol': 0.4},
            'vui_ve': {'dopamine': 0.3, 'endorphins': 0.2}, 'lo_au': {'cortisol': 0.5, 'adrenaline': 0.3},
            'an_ui': {'oxytocin': 0.4, 'serotonin': 0.3}, 'uc_che': {'GABA': 0.4}
        }
        
        self.current_emotional_state = defaultdict(float)
        self.last_recall_time = 0
        self.RECALL_COOLDOWN = 5.0 # Thời gian nghỉ 5 giây
        logger.info("ExperienceEncoder initialized.")      
    def _build_autoencoder(self, input_dim, embedding_dim):
        """Xây dựng Autoencoder với kích thước đầu vào linh hoạt"""
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, embedding_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128), nn.ReLU(),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.Linear(64, embedding_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(embedding_dim, 64), nn.ReLU(),
                    nn.Linear(64, 128), nn.ReLU(),
                    nn.Linear(128, input_dim)
                )
            def forward(self, x):
                return self.encoder(x), self.decoder(self.encoder(x))
        return Autoencoder(input_dim, embedding_dim)
    def _normalize_ampha(self, ampha_data):
        """Chuẩn hóa dữ liệu ampha thành vector 8 chiều"""
        vector = np.zeros(self.AMPHA_VECTOR_SIZE)
        if not ampha_data: return vector
        emotion_index = {'HY':0, 'no':1, 'ai':2, 'O':3, 'BI':4, 'A':5, 'lac':6, 'Cu':7}
        for component in ampha_data.get('components', []):
            if component in emotion_index:
                idx = emotion_index[component]
                vector[idx] = max(vector[idx], ampha_data.get('intensity', 0))
        return vector
    def encode_experience(self, sensory_data, attention_data, hormone_data, ampha_data):
        """
        Mã hóa trải nghiệm thành vector nhúng sử dụng Autoencoder
        
        Parameters:
        - sensory_data: Dữ liệu giác quan đã xử lý (dict)
        - attention_data: Thông tin attention (score, status, types) (tuple)
        - hormone_data: Mức hormone hiện tại (dict)
        
        Returns:
        - embedding: Vector nhúng đại diện cho trải nghiệm
        """
    def encode_experience(self, sensory_data, attention_data, hormone_data, ampha_data):
        """Mã hóa trải nghiệm, bao gồm cả ampha, thành vector nhúng"""
        norm_sensory = self._normalize_sensory(sensory_data)
        norm_attention = self._normalize_attention(attention_data)
        norm_hormones = self._normalize_hormones(hormone_data)
        ampha_vector = self._normalize_ampha(ampha_data)
        
        feature_vector = np.concatenate([
            norm_sensory, norm_attention, norm_hormones, ampha_vector
        ])
        
        expected_dim = self.BASE_FEATURE_SIZE + self.AMPHA_VECTOR_SIZE
        if len(feature_vector) != expected_dim:
            logger.error(f"Dimension mismatch! Expected {expected_dim}, got {len(feature_vector)}")
            # Pad or truncate to handle mismatch
            feature_vector = np.resize(feature_vector, expected_dim)

        with torch.no_grad():
            features_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            embedding, _ = self.encoder(features_tensor)
        return embedding.squeeze(0).numpy()

    def _normalize_sensory(self, sensory_data):
        """Chuẩn hóa dữ liệu giác quan (16 đặc trưng)"""
        cam = sensory_data.get('camera', {})
        mic = sensory_data.get('mic', {})
        return np.array([
            cam.get('brightness', 0.5), cam.get('contrast', 0.5), cam.get('motion', 0.0),
            cam.get('face_detected', False), cam.get('hue_R', 0), cam.get('hue_G', 0),
            cam.get('hue_B', 0), cam.get('saturation', 0.5), cam.get('object_size', 0),
            cam.get('object_distance', 1.0), mic.get('volume_dB', 60)/100, mic.get('complexity', 0.5),
            len(sensory_data.get('smell', [])), len(sensory_data.get('taste', [])),
            len(sensory_data.get('force', [])), sensory_data.get('temperature_C', 37)/40
        ])
    def _normalize_attention(self, attention_data):
        """Chuẩn hóa thông tin attention (5 đặc trưng)"""
        score, status, types = attention_data
        status_map = {"high attend": 1.0, "moderate attend": 0.7, "neutral": 0.5, "unattend": 0.3, "strong unattend": 0.1}
        return np.array([
            score, status_map.get(status, 0.5), len(types)/10.0,
            1.0 if 'face_focused' in types else 0.0, 1.0 if 'motion_tracking' in types else 0.0
        ])
    
    def _normalize_hormones(self, hormone_data):
        """Chuẩn hóa dữ liệu hormone (7 đặc trưng)"""
        return np.array([
            hormone_data.get(h, 0) for h in ['oxytocin', 'cortisol', 'dopamine', 'serotonin', 'adrenaline', 'endorphins', 'melatonin']
        ])

    def _determine_primary_emotion(self, hormone_data):
        """Xác định cảm xúc chính từ hormone"""
        emotion_scores = defaultdict(float)
        if hormone_data.get('oxytocin', 0) > 0.6: emotion_scores['an_toan'] += hormone_data['oxytocin']
        if hormone_data.get('cortisol', 0) > 0.5: emotion_scores['cang_thang'] += hormone_data['cortisol']
        if hormone_data.get('dopamine', 0) > 0.5: emotion_scores['vui_ve'] += hormone_data['dopamine']
        if hormone_data.get('serotonin', 0) > 0.6: emotion_scores['hai_long'] += hormone_data['serotonin']
        return max(emotion_scores, key=emotion_scores.get) if emotion_scores else 'trung_tinh'

    def _supplement_emotion_from_attention(self, primary_emotion, attention_types):
        """Bổ sung nhãn cảm xúc từ attention types"""
        emotions = {primary_emotion}
        if 'startled_by_loud_noise' in attention_types: emotions.add('so_hai')
        if 'comforted_by_caregiver' in attention_types: emotions.add('an_ui')
        if 'sweet_taste_seeking' in attention_types: emotions.add('thich_thu')
        return emotions

    def update_emotional_state(self, new_attention_types, hormone_data, delta_time=1.0, emotion_delta=None):
        """Cập nhật trạng thái cảm xúc nội tại"""
        for emotion in list(self.current_emotional_state.keys()):
            self.current_emotional_state[emotion] *= (0.95 ** delta_time)
            if self.current_emotional_state[emotion] < 0.01:
                del self.current_emotional_state[emotion]

        if emotion_delta:
            for emotion in emotion_delta.get('suppress', []): self.current_emotional_state[emotion] = 0
            for emotion, boost in emotion_delta.get('boost', {}).items():
                self.current_emotional_state[emotion] = min(1.0, self.current_emotional_state.get(emotion, 0) + boost)

        primary_emotion = self._determine_primary_emotion(hormone_data)
        self.current_emotional_state[primary_emotion] = min(1.0, self.current_emotional_state.get(primary_emotion, 0) + 0.1)
        
        return self.current_emotional_state

    def store_experience(self, embedding, hormone_data, attention_types, score, ampha_data, personality_profile, delta_vector):
        """Lưu trữ trải nghiệm, bao gồm cả ngữ cảnh tính cách và vector delta"""
        primary_emotion = self._determine_primary_emotion(hormone_data)
        emotion_tags = self._supplement_emotion_from_attention(primary_emotion, attention_types)
        
        dynamic_weight = score + hormone_data.get('cortisol', 0) * 0.3 + hormone_data.get('adrenaline', 0) * 0.2
        dynamic_weight = np.clip(dynamic_weight, 0.1, 1.0)
        
        experience = {
            'embedding': embedding,
            'emotion_tags': emotion_tags,
            'hormone_data': hormone_data.copy(),
            'weight': dynamic_weight,
            'ampha_data': ampha_data,
            'personality_context': personality_profile.copy(),
            'delta_vector': delta_vector.copy()   # LƯU VECTOR DELTA
        }
        self.short_term_memory.append(experience)
        
        if dynamic_weight > self.long_term_thresh:
            self.long_term_memory.append(embedding)
            self.long_term_weights.append(dynamic_weight)
            self.long_term_emotions.append(emotion_tags)
            self.long_term_personality_contexts.append(personality_profile.copy())
            self.long_term_delta_vectors.append(delta_vector.copy())  # LƯU DELTA VÀO LTM
            self.rebuild_counter += 1
            logger.info(f"Experience moved to LTM with weight {dynamic_weight:.2f} and personality context.")
            if self.rebuild_counter >= self.REBUILD_INTERVAL:
                self._rebuild_kdtree()

    def retrieve_related_experiences(self, query_embedding, current_personality_profile,current_delta, k=5, memory_type='both'):
        """
        Truy xuất trải nghiệm liên quan, có tính đến sự tương đồng về tính cách.
        """
        results = []
        recall_triggered = False
        can_recall = (time.time() - self.last_recall_time) > self.RECALL_COOLDOWN

        # Truy vấn STM
        for exp in self.short_term_memory:
            distance = np.linalg.norm(exp['embedding'] - query_embedding)
            results.append({
                'source': 'short_term', 'distance': distance, 'weight': exp['weight'],
                'emotion_tags': exp['emotion_tags'], 'ampha_data': exp.get('ampha_data'),
                'personality_match': self._calculate_personality_similarity(
                    current_personality_profile, exp.get('personality_context', {})
                )
            })

        # Truy vấn LTM
        if memory_type in ['long', 'both'] and self.kdtree and len(self.long_term_memory) > 0:
            actual_k = min(k, len(self.long_term_memory))
            distances, indices = self.kdtree.query([query_embedding], k=actual_k)
            for i, idx in enumerate(indices[0]):
                results.append({
                    'source': 'long_term', 'distance': distances[0][i], 'weight': self.long_term_weights[idx],
                    'emotion_tags': self.long_term_emotions[idx], 'ampha_data': None, # Ampha không lưu chi tiết trong LTM
                    'personality_match': self._calculate_personality_similarity(
                        current_personality_profile, self.long_term_personality_contexts[idx]
                    )
                })

        # Sắp xếp kết quả: ưu tiên khoảng cách thấp, sau đó là tương đồng tính cách cao
        # Sắp xếp kết quả dựa trên similarity + delta similarity
        results.sort(key=lambda x: (
            x['distance'], 
            -x['personality_match'],
            -self._delta_similarity(current_delta, x.get('delta_vector', []))
        ))
        
        # Kích hoạt hồi tưởng cảm xúc
        if can_recall and results and results[0]['distance'] < self.emotion_recall_long:
            recall_triggered = True
            self.last_recall_time = time.time()
            logger.info(f"EMOTION RECALL TRIGGERED for tags: {results[0]['emotion_tags']}")
            # self._trigger_emotion_recall_hormones(results[0]['emotion_tags']) # (Optional)

        return results[:k], recall_triggered
    def _delta_similarity(self, delta1, delta2):
        """Tính độ tương đồng giữa hai vector delta"""
        if len(delta1) != len(delta2) or len(delta1) == 0:
            return 0.0
            
        # Sử dụng cosine similarity
        dot_product = np.dot(delta1, delta2)
        norm1 = np.linalg.norm(delta1)
        norm2 = np.linalg.norm(delta2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)    
    def _calculate_personality_similarity(self, profile1, profile2):
        """Tính toán độ tương đồng cosine giữa hai hồ sơ tính cách."""
        if not profile1 or not profile2: return 0.0
        
        all_keys = set(profile1.keys()) | set(profile2.keys())
        vec1 = np.array([profile1.get(k, 0) for k in all_keys])
        vec2 = np.array([profile2.get(k, 0) for k in all_keys])
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0: return 0.0
        return dot_product / (norm1 * norm2)
    
    def _trigger_emotion_recall_hormones(self, emotion_tags):
        """Kích hoạt thay đổi hormone khi hồi tưởng cảm xúc"""
        hormone_changes = {}
        emotion_to_hormone = {
            'an_toan': {'oxytocin': 0.3},
            'cang_thang': {'cortisol': 0.4},
            'vui_ve': {'dopamine': 0.3, 'endorphins': 0.2},
            'lo_au': {'cortisol': 0.5, 'adrenaline': 0.3},
            'an_ui': {'oxytocin': 0.4, 'serotonin': 0.3},
            'uc_che': {'GABA': 0.4}
        }
        
        for emotion in emotion_tags:
            if emotion in emotion_to_hormone:
                for hormone, change in emotion_to_hormone[emotion].items():
                    hormone_changes[hormone] = hormone_changes.get(hormone, 0) + change
        
        if hormone_changes:
            print(f"KÍCH HOẠT HỒI TƯỞNG: {emotion_tags} → Hormone: {hormone_changes}")

    def consolidate_memory(self):
        """Củng cố bộ nhớ, chuyển từ STM sang LTM."""
        logger.info("--- Consolidating memory ---")
        moved_count = 0
        stm_copy = list(self.short_term_memory)
        for exp in stm_copy:
            if exp.get('weight', 0) > 0.8:
                self.long_term_memory.append(exp['embedding'])
                self.long_term_weights.append(exp['weight'])
                self.long_term_emotions.append(exp['emotion_tags'])
                self.long_term_personality_contexts.append(exp['personality_context'])
                moved_count += 1
                self.short_term_memory.remove(exp)
        
        if moved_count > 0:
            logger.info(f"Moved {moved_count} experiences from STM to LTM.")
            self._rebuild_kdtree()
    def _rebuild_kdtree(self):
        """Xây dựng lại KD-Tree cho LTM."""
        if self.long_term_memory:
            self.kdtree = KDTree(np.array(self.long_term_memory))
            self._prune_long_term_memory() # Dọn dẹp sau khi xây lại
        else:
            self.kdtree = None            
    def _prune_long_term_memory(self):
        """Loại bỏ ký ức LTM có trọng số quá thấp."""
        if not self.long_term_memory: return
        
        prune_threshold = 0.05
        indices_to_keep = [i for i, w in enumerate(self.long_term_weights) if w > prune_threshold]
        
        if len(indices_to_keep) < len(self.long_term_weights):
            self.long_term_memory = [self.long_term_memory[i] for i in indices_to_keep]
            self.long_term_weights = [self.long_term_weights[i] for i in indices_to_keep]
            self.long_term_emotions = [self.long_term_emotions[i] for i in indices_to_keep]
            self.long_term_personality_contexts = [self.long_term_personality_contexts[i] for i in indices_to_keep]
            logger.info(f"Pruned LTM, removed {len(self.long_term_weights) - len(indices_to_keep)} weak memories.")
            # Xây dựng lại tree sau khi dọn dẹp
            if self.long_term_memory:
                self.kdtree = KDTree(np.array(self.long_term_memory))
            else:
                self.kdtree = None

    def update_weights_based_on_hormones(self, current_hormones, delta_time):
        """
        Cập nhật trọng số ký ức dựa trên hormone hiện tại và thời gian trôi qua
        - Áp dụng phân rã theo thời gian (time-based decay)
        - Tăng cường ký ức tương đồng hormone
        """
        # Tính toán hệ số phân rã dựa trên delta_time
        decay_factor = self.decay_factor ** delta_time
        
        # 1. Cập nhật bộ nhớ ngắn hạn (STM)
        for exp in self.short_term_memory:
            # Áp dụng phân rã thời gian
            exp['weight'] *= decay_factor
            
            # Tính toán tương đồng hormone nếu có dữ liệu hormone lưu trữ
            if 'hormone_data' in exp:
                hormone_similarity = self._calculate_hormone_similarity(
                    current_hormones, 
                    exp['hormone_data']
                )
                
                # Tăng cường ký ức có hormone tương đồng
                if hormone_similarity > 0.7:
                    boost = 0.2 * hormone_similarity
                    exp['weight'] = min(1.0, exp['weight'] + boost)
        
        # 2. Cập nhật bộ nhớ dài hạn (LTM)
        for i in range(len(self.long_term_weights)):
            # Áp dụng phân rã thời gian
            self.long_term_weights[i] *= decay_factor
            
            # Kiểm tra nếu hormone hiện tại cao và phù hợp với cảm xúc của ký ức
            for hormone, value in current_hormones.items():
                if value > 0.8 and i < len(self.long_term_emotions):
                    emotion_tags = self.long_term_emotions[i]
                    
                    # Tìm hormone liên quan đến cảm xúc của ký ức
                    for emotion in emotion_tags:
                        if emotion in self.emotion_to_hormone:
                            if hormone in self.emotion_to_hormone[emotion]:
                                # Tăng cường ký ức phù hợp
                                self.long_term_weights[i] = min(1.0, self.long_term_weights[i] + 0.3)
                                break
        
        # Loại bỏ ký ức có trọng số quá thấp
        self._prune_long_term_memory()
        
        # Tăng bộ đếm xây dựng lại KD-Tree
        self.rebuild_counter += 1
        if self.rebuild_counter >= self.REBUILD_INTERVAL:
            self._rebuild_kdtree()
            self.rebuild_counter = 0

        
    def _calculate_hormone_similarity(self, current, stored):
        """Tính độ tương đồng giữa hormone hiện tại và hormone lưu trữ"""
        hormones = set(current.keys()) | set(stored.keys())
        vec_current = np.array([current.get(h, 0) for h in hormones])
        vec_stored = np.array([stored.get(h, 0) for h in hormones])
        
        dot_product = np.dot(vec_current, vec_stored)
        norm_current = np.linalg.norm(vec_current)
        norm_stored = np.linalg.norm(vec_stored)
        
        if norm_current == 0 or norm_stored == 0:
            return 0.0
            
        return dot_product / (norm_current * norm_stored)

    def update_emotion_associations(self, embedding, observed_hormones, ampha_data):
        """Cập nhật mạng neural liên kết cảm xúc"""
        hormone_vec = torch.zeros(len(self.EMOTION_MAPPING))
        for i, h in enumerate(self.EMOTION_MAPPING.keys()):
            hormone_vec[i] = 1.0 if observed_hormones.get(h, 0) > 0.5 else 0.0
        ampha_vector = self._normalize_ampha(ampha_data)            
        inputs = torch.cat([
            torch.tensor(embedding, dtype=torch.float32), 
            hormone_vec.float(), 
            torch.tensor(ampha_vector, dtype=torch.float32)
        ])
        outputs = self.emotion_net(inputs.float())
        loss = F.binary_cross_entropy(outputs, hormone_vec)
        
        self.emotion_optimizer.zero_grad()
        loss.backward()
        self.emotion_optimizer.step()
        
        return loss.item()

    def predict_emotion(self, embedding, ampha_data=None):
        """Dự đoán cảm xúc với thông tin ampha"""
        with torch.no_grad():
            hormone_vec = torch.zeros(self.EMOTION_MAPPING_SIZE)
            
            # Thêm dữ liệu ampha nếu có
            if ampha_data:
                ampha_vector = self._normalize_ampha(ampha_data)
                inputs = torch.cat([
                    torch.tensor(embedding, dtype=torch.float32), 
                    hormone_vec.float(), 
                    torch.tensor(ampha_vector, dtype=torch.float32)
                ])
            else: # Nếu không có ampha, dùng vector 0
                ampha_vector = np.zeros(self.AMPHA_VECTOR_SIZE)
                inputs = torch.cat([
                    torch.tensor(embedding, dtype=torch.float32), 
                    hormone_vec.float(), 
                    torch.tensor(ampha_vector, dtype=torch.float32)
                ])
            
            outputs = self.emotion_net(inputs)
            return {e: outputs[i].item() for i, e in enumerate(self.EMOTION_MAPPING.values())}


class PreferenceMemorySystem:
    def __init__(self):
        self.emotion_anchors = defaultdict(list)
        self.preferences = defaultdict(float)
        
    def create_emotion_anchor(self, ampha_key, sensory_pattern, strength=1.0):
        self.emotion_anchors[ampha_key].append({
            'sensory_pattern': sensory_pattern,
            'strength': strength,
            'last_activated': time.time()
        })
        
    def reinforce_anchor(self, ampha_key, sensory_pattern, delta_strength=0.1):
        for anchor in self.emotion_anchors.get(ampha_key, []):
            if self._similarity(anchor['sensory_pattern'], sensory_pattern) > 0.7:
                anchor['strength'] = min(1.0, anchor['strength'] + delta_strength)
                anchor['last_activated'] = time.time()
                return True
        return False
        
    def update_preference(self, sensory_pattern, reward):
        pattern_key = self._pattern_to_key(sensory_pattern)
        self.preferences[pattern_key] = self.preferences.get(pattern_key, 0) + reward
        
    def get_preferred_stimuli(self, ampha_key=None):
        if ampha_key:
            return [anchor['sensory_pattern'] for anchor in self.emotion_anchors.get(ampha_key, [])]
        else:
            return sorted(self.preferences.items(), key=lambda x: x[1], reverse=True)[:5]
            
    def _similarity(self, pattern1, pattern2):
        return 0.8
        
    def _pattern_to_key(self, pattern):
        return str(hash(frozenset(pattern.items())))  