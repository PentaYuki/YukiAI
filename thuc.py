# thuc.py
import torch
from sllm_bridge import sLLM_reply

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from collections import deque, defaultdict

logger = logging.getLogger('ThucModule')

class ThucConsciousnessModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, active_intent_dim, mental_disposition_dim):
        """
        Mô-đun thức (Consciousness Layer) tích hợp LSTM và RL
        - Tách rõ Ý tâm (Mental Disposition) và Ý động (Active Intent)
        - Quản lý dòng tâm thức theo thời gian
        
        Args:
            input_dim: Kích thước vector đầu vào (nội tại + môi trường + bối cảnh)
            hidden_dim: Kích thước hidden state của LSTM
            active_intent_dim: Số lượng ý định hành động có thể có
            mental_disposition_dim: Số chiều vector ý tâm
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_active_intent = nn.Linear(hidden_dim, active_intent_dim)
        self.fc_mental_disposition = nn.Linear(hidden_dim, mental_disposition_dim)
        self.hidden_state = None
        self.cell_state = None
        
        # Theo dõi lịch sử tâm thức
        self.consciousness_stream = deque(maxlen=100)
        self.action_history = deque(maxlen=20)
        
        # Bộ nhớ ngắn hạn cho trạng thái ẩn
        self.hidden_state_memory = deque(maxlen=5)
        
        logger.info("Khởi tạo ThucConsciousnessModule: input_dim=%d, hidden_dim=%d", input_dim, hidden_dim)
        
    def forward(self, x, reset_flag=False):
        if reset_flag:
            self._reset_consciousness()
            logger.warning("RESET Ý THỨC do sự kiện đặc biệt")
        
        # Đảm bảo tensor đầu vào có đúng kích thước 3D: (batch_size, seq_len, input_dim)
        if x.dim() == 1:
            # Nếu là vector 1D, thêm chiều batch và sequence
            x = x.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, input_dim)
        elif x.dim() == 2:
            # Nếu là ma trận 2D, thêm chiều sequence
            x = x.unsqueeze(1)  # shape: (batch_size, 1, input_dim)
        
        if self.hidden_state is None:
            h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
            c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
        else:
            h0, c0 = self.hidden_state, self.cell_state
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        self.hidden_state, self.cell_state = hn, cn
        
        # Lấy output tại bước cuối cùng của sequence
        last_output = out[:, -1, :]
        
        # Tách thành Ý tâm và Ý động
        mental_disposition = self.fc_mental_disposition(last_output)
        active_intent = self.fc_active_intent(last_output)
        
        # Ghi lại trạng thái ẩn
        self.hidden_state_memory.append(hn.detach().clone())
        
        logger.debug("Xử lý dòng tâm thức | Ý tâm: %s | Ý động: %s", 
                     mental_disposition.shape, active_intent.shape)
        
        return active_intent, mental_disposition
    def _reset_consciousness(self):
        """Thiết lập lại dòng tâm thức"""
        self.hidden_state = None
        self.cell_state = None
        self.hidden_state_memory.clear()
        
    def attend_to_hidden_state(self):
        """Cơ chế chú ý giữa các trạng thái ẩn"""
        if len(self.hidden_state_memory) < 2:
            return None
            
        # Tính attention giữa các hidden state
        states = torch.stack(list(self.hidden_state_memory))
        attention_weights = F.softmax(torch.matmul(states, states[-1].t()), dim=0)
        attended_state = torch.sum(attention_weights * states, dim=0)
        
        logger.debug("Ứng dụng attention giữa %d trạng thái ẩn", len(self.hidden_state_memory))
        return attended_state

class RewardEvaluator:
    def __init__(self, 
                 short_term_weights,
                 long_term_weights,
                 delusion_penalty=0.5,
                 long_term_window=30):
        """
        Đánh giá phần thưởng tách biệt ngắn hạn/dài hạn
        - short_term_weights: Trọng số phần thưởng ngắn hạn
        - long_term_weights: Trọng số phần thưởng dài hạn
        - delusion_penalty: Hình phạt cho hành vi Si (ảo tưởng)
        - long_term_window: Cửa sổ thời gian cho phần thưởng dài hạn
        """
        self.short_term_weights = short_term_weights
        self.long_term_weights = long_term_weights
        self.delusion_penalty = delusion_penalty
        self.long_term_window = long_term_window
        
        # Theo dõi lịch sử hormone
        self.hormone_history = defaultdict(lambda: deque(maxlen=long_term_window))
        
        # Theo dõi hành vi thất bại (cho hình phạt Si)
        self.failed_actions = defaultdict(int)
        
        logger.info("Khởi tạo RewardEvaluator")

    def compute_reward(self, current_hormones, action_taken, action_success):
        """
        Tính toán phần thưởng tổng hợp:
        - Phần thưởng ngắn hạn: Phản ứng tức thì với hormone
        - Phần thưởng dài hạn: Duy trì trạng thái tích cực
        - Hình phạt Si: Trừng phạt hành vi cố chấp thất bại
        """
        # Cập nhật lịch sử hormone
        for h, v in current_hormones.items():
            self.hormone_history[h].append(v)
        
        # 1. Phần thưởng ngắn hạn
        short_term_reward = 0
        for hormone, weight in self.short_term_weights.items():
            if hormone in current_hormones:
                short_term_reward += weight * current_hormones[hormone]
        
        # 2. Phần thưởng dài hạn
        long_term_reward = 0
        for hormone, weight in self.long_term_weights.items():
            if hormone in self.hormone_history:
                avg_level = sum(self.hormone_history[hormone]) / len(self.hormone_history[hormone])
                if avg_level > 0.6:  # Chỉ thưởng khi duy trì trạng thái tốt
                    long_term_reward += weight * avg_level
        
        # 3. Hình phạt Si (ảo tưởng, cố chấp)
        delusion_penalty = 0
        if not action_success:
            self.failed_actions[action_taken] += 1
            if self.failed_actions[action_taken] > 2:  # Phạt nếu lặp lại thất bại
                delusion_penalty = -self.delusion_penalty * self.failed_actions[action_taken]
                logger.warning(f"HÌNH PHẠT SI: {action_taken} thất bại {self.failed_actions[action_taken]} lần")
        else:
            self.failed_actions[action_taken] = 0  # Reset khi thành công
        
        total_reward = short_term_reward + long_term_reward + delusion_penalty
        
        logger.debug(f"PHẦN THƯỞNG: ngắn={short_term_reward:.2f}, dài={long_term_reward:.2f}, phạt={delusion_penalty:.2f}, tổng={total_reward:.2f}")
        return total_reward

# Ví dụ sử dụng trong hệ thống
if __name__ == "__main__":
    # Cấu hình logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Khởi tạo module thức
    thuc = ThucConsciousnessModule(
        input_dim=128, 
        hidden_dim=64,
        active_intent_dim=8,     # 8 loại ý định hành động
        mental_disposition_dim=5 # 5 chiều ý tâm
    )
    
    # Giả lập dữ liệu đầu vào
    input_state = torch.randn(128)
    
    # Xử lý dòng tâm thức
    active_intent, mental_disposition = thuc(input_state)
    
    print("Ý động (Active Intent):", active_intent.shape)
    print("Ý tâm (Mental Disposition):", mental_disposition.shape)
    response = sLLM_reply("Mức dopamine hiện tại là 0.8, cortisol 0.2. Yuki cảm thấy thế nào?")
    print(response)
