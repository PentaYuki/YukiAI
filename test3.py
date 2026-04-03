# test4.py (Nâng cấp Trực quan & Phân tích)
try:
    import time
    import numpy as np
    import random
    import logging
    from collections import defaultdict
    import cv2
    import sys
    import os

    # Thêm đường dẫn project vào sys.path để import dễ dàng
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # --- Import tất cả các lớp và hằng số ---
    from sac.sac import SENSOR_WEIGHTS, SAFETY_THRESHOLDS, SMELL_COMPOUNDS, TASTE_COMPOUNDS, SENSITIVE_AREAS
    from attendcore.corehocmon import ATTENTION_HORMONE_MAP, ATTENTION_STATUS_HORMONES
    from sac.Vision import VisionProcessor
    from sac.Audio import AudioProcessor
    from attendcore.corehocmon2 import HormoneDynamics
    from memory.expAttend import ExperienceEncoder
    from attendcore.corephysiological import PhysiologicalCenter
    from tho.thoemotion import EmotionIntegrationModule
    from tho.thohoocmon import EmotionHarmonyBridge
    from tuong import BabyAttentionModel

except ImportError as e:
    print(f"Lỗi import: {e}. Vui lòng kiểm tra lại cấu trúc thư mục và sys.path.")
    exit(1)


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger('Simulation')

# ==============================================================================
# === MODULE TRỰC QUAN HÓA MỚI ===
# ==============================================================================
class Visualizer:
    """Lớp chuyên dụng để vẽ thông tin lên frame hình ảnh."""
    def __init__(self, frame_width):
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.6
        self.FONT_THICKNESS = 1
        self.PANEL_WIDTH = 350
        self.FRAME_WIDTH = frame_width
        
        # Màu sắc
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_GREEN = (100, 255, 100)
        self.COLOR_RED = (100, 100, 255)
        self.COLOR_BLUE = (255, 150, 100)
        self.COLOR_YELLOW = (0, 220, 255)
        self.COLOR_PURPLE = (220, 100, 220)

    def _draw_panel(self, frame, title, data, start_y, color):
        """Vẽ một panel thông tin."""
        cv2.putText(frame, title, (10, start_y), self.FONT, 0.7, color, 2)
        for i, (label, value_str) in enumerate(data.items()):
            y_pos = start_y + 30 + i * 25
            cv2.putText(frame, f"{label}: {value_str}", (15, y_pos), self.FONT, self.FONT_SCALE, self.COLOR_WHITE, self.FONT_THICKNESS)
    
    def _draw_bar(self, frame, x, y, label, value, max_value, color, bar_width=150):
        """Vẽ một thanh bar biểu diễn giá trị."""
        # Chuẩn hóa giá trị về 0-1
        normalized_value = np.clip(value / max_value, 0, 1)
        
        # Vẽ label
        cv2.putText(frame, label, (x, y), self.FONT, self.FONT_SCALE, self.COLOR_WHITE, self.FONT_THICKNESS)
        
        # Vẽ nền của thanh bar
        cv2.rectangle(frame, (x + 110, y - 12), (x + 110 + bar_width, y + 5), (50, 50, 50), -1)
        # Vẽ phần giá trị của thanh bar
        fill_width = int(normalized_value * bar_width)
        cv2.rectangle(frame, (x + 110, y - 12), (x + 110 + fill_width, y + 5), color, -1)
        cv2.putText(frame, f"{value:.2f}", (x + 115 + bar_width, y + 5), self.FONT, 0.5, self.COLOR_WHITE, 1)

    def display(self, frame, sim_data):
        """Hiển thị tất cả thông tin lên frame."""
        # Panel Trạng Thái Tức Thời (Delta & Hormones)
        y_offset = 30
        cv2.putText(frame, "TRANG THAI TUC THOI (Delta)", (10, y_offset), self.FONT, 0.7, self.COLOR_YELLOW, 2)
        y_offset += 30
        delta_vec = sim_data.get('delta_vector', {})
        self._draw_bar(frame, 15, y_offset, "Valence", delta_vec.get('valence', 0), 1.0, self.COLOR_GREEN if delta_vec.get('valence', 0) > 0 else self.COLOR_RED)
        y_offset += 30
        self._draw_bar(frame, 15, y_offset, "Arousal", delta_vec.get('arousal', 0), 1.0, self.COLOR_PURPLE)
        y_offset += 30
        self._draw_bar(frame, 15, y_offset, "Security", delta_vec.get('security', 0), 1.0, self.COLOR_BLUE)
        
        y_offset += 40
        cv2.putText(frame, "Hormone Chinh", (10, y_offset), self.FONT, 0.7, self.COLOR_YELLOW, 2)
        y_offset += 30
        hormones = sim_data.get('hormone_summary', {})
        self._draw_bar(frame, 15, y_offset, "Cortisol", hormones.get('cortisol', 0), 2.0, self.COLOR_RED)
        y_offset += 30
        self._draw_bar(frame, 15, y_offset, "Oxytocin", hormones.get('oxytocin', 0), 2.0, self.COLOR_BLUE)
        y_offset += 30
        self._draw_bar(frame, 15, y_offset, "Dopamine", hormones.get('dopamine', 0), 2.0, self.COLOR_GREEN)

        # Panel Tính Cách & Ampha
        x_offset = self.FRAME_WIDTH - self.PANEL_WIDTH
        y_offset = 30
        cv2.putText(frame, "TINH CACH & AMPHA", (x_offset, y_offset), self.FONT, 0.7, self.COLOR_YELLOW, 2)
        y_offset += 30
        ampha_key = sim_data.get('ampha_key', 'None')
        ampha_intensity = sim_data.get('ampha_intensity', 0)
        cv2.putText(frame, f"Ampha: {ampha_key} (I:{ampha_intensity:.2f})", (x_offset + 5, y_offset), self.FONT, self.FONT_SCALE, self.COLOR_WHITE, self.FONT_THICKNESS)

        y_offset += 40
        cv2.putText(frame, "Tinh Cach (Top 3)", (x_offset, y_offset), self.FONT, 0.7, self.COLOR_YELLOW, 2)
        y_offset += 30
        personality = sim_data.get('personality', {})
        sorted_personality = sorted(personality.items(), key=lambda item: item[1], reverse=True)
        for i, (trait, value) in enumerate(sorted_personality[:3]):
            self._draw_bar(frame, x_offset + 5, y_offset, trait, value, 1.0, self.COLOR_GREEN, bar_width=100)
            y_offset += 30
            
        # Panel Thông tin chung
        info_y = frame.shape[0] - 100
        cv2.putText(frame, f"KICH BAN: {sim_data.get('scenario_name', 'N/A')}", (10, info_y), self.FONT, self.FONT_SCALE, self.COLOR_YELLOW, self.FONT_THICKNESS)
        cv2.putText(frame, f"CHU Y: {sim_data.get('status', 'N/A').upper()} ({sim_data.get('score', 0):.2f})", (10, info_y + 25), self.FONT, self.FONT_SCALE, self.COLOR_WHITE, self.FONT_THICKNESS)
        physio = sim_data.get('physiological_state', {})
        cv2.putText(frame, f"SINH LY: Tim={int(physio.get('heart_rate', 0))} | Tho={int(physio.get('breathing_rate', 0))}", (10, info_y + 50), self.FONT, self.FONT_SCALE, self.COLOR_WHITE, self.FONT_THICKNESS)


# ==============================================================================
# === SCENARIO MANAGER NÂNG CẤP ===
# ==============================================================================
class ScenarioManager:
    """Quản lý các kịch bản kiểm thử theo từng giai đoạn."""
    def __init__(self, cycle_length):
        self.cycle_length = cycle_length
        self.scenarios = [
            self._scenario_neutral_reset,      # Bắt đầu với trạng thái nghỉ
            self._scenario_positive_formation,
            self._scenario_test_after_positive,
            self._scenario_neutral_reset,      # Nghỉ giữa hiệp
            self._scenario_negative_formation,
            self._scenario_test_after_negative,
        ]
        self.current_scenario_index = 0
        self.scenario_start_frame = 0

    def update(self, frame_count):
        """Chuyển kịch bản nếu cần."""
        if frame_count - self.scenario_start_frame >= self.cycle_length:
            self.current_scenario_index = (self.current_scenario_index + 1) % len(self.scenarios)
            self.scenario_start_frame = frame_count
            logger.warning(f"--- CHUYEN SANG KICH BAN MOI: {self.get_current_scenario_name()} ---")

    def get_current_sensory_input(self):
        """Lấy dữ liệu giác quan cho kịch bản hiện tại."""
        return self.scenarios[self.current_scenario_index]()

    def get_current_scenario_name(self):
        """Lấy tên của kịch bản hiện tại."""
        return self.scenarios[self.current_scenario_index].__name__.replace("_scenario_", "").upper()
    
    def _scenario_neutral_reset(self):
        """Giai đoạn nghỉ: Kích thích rất nhẹ để AI trở về trạng thái cân bằng."""
        return {
            'taste': [], 'force': [], 'smell': []
        }
        
    def _scenario_positive_formation(self):
        """Giai đoạn 1: Liên tục tạo kích thích tích cực để hình thành tính cách 'ailac'."""
        return {
            'taste': [{'compound': 'lactose', 'concentration_mg_dl': 50.0}],
            'force': [{'body_part': 'hand', 'force_N': 1.0, 'is_tickle': False}],
            'smell': [{'compound': 'vanillin', 'concentration_ppm': 0.2}]
        }

    def _scenario_test_after_positive(self):
        """Giai đoạn 2: Kích thích tiêu cực nhẹ để xem AI 'lạc quan' phản ứng ra sao."""
        return {
            'taste': [{'compound': 'caffeine', 'concentration_mg_dl': 10.0}],
            'force': [{'body_part': 'hand', 'force_N': 2.5, 'is_tickle': False}],
            'smell': [{'compound': 'isovaleric_acid', 'concentration_ppm': 0.1}]
        }
        
    def _scenario_negative_formation(self):
        """Giai đoạn 3: Liên tục tạo kích thích tiêu cực để hình thành tính cách 'Cuno'."""
        return {
            'taste': [{'compound': 'quinine', 'concentration_mg_dl': 25.0}],
            'force': [{'body_part': 'stomach', 'force_N': 3.5, 'is_tickle': False}],
            'smell': [{'compound': 'hexanoic_acid', 'concentration_ppm': 0.3}]
        }

    def _scenario_test_after_negative(self):
        """Giai đoạn 4: Kích thích tích cực để xem AI 'tiêu cực' phản ứng ra sao."""
        return {
            'taste': [{'compound': 'lactose', 'concentration_mg_dl': 50.0}],
            'force': [{'body_part': 'hand', 'force_N': 1.0, 'is_tickle': False}],
            'smell': [{'compound': 'vanillin', 'concentration_ppm': 0.2}]
        }


def main():
    logger.info("--- BẮT ĐẦU MÔ PHỎNG MÔ HÌNH CHÚ Ý CỦA BÉ (BẢN NÂNG CẤP TRỰC QUAN) ---")

    # --- 1. KHỞI TẠO CÁC THÀNH PHẦN ---
    AGE_MONTHS = 12
    config = {
        'SENSOR_WEIGHTS': SENSOR_WEIGHTS, 'SAFETY_THRESHOLDS': SAFETY_THRESHOLDS,
        'SMELL_COMPOUNDS': SMELL_COMPOUNDS, 'TASTE_COMPOUNDS': TASTE_COMPOUNDS,
        'SENSITIVE_AREAS': SENSITIVE_AREAS, 'ATTENTION_HORMONE_MAP': ATTENTION_HORMONE_MAP,
        'ATTENTION_STATUS_HORMONES': ATTENTION_STATUS_HORMONES, 'STATUS_IMPACT_FACTOR': 1.2
    }

    hormone_system = HormoneDynamics()
    memory_system = ExperienceEncoder(short_term_capacity=20, long_term_capacity=1000)
    physio_center = PhysiologicalCenter()
    vision_processor = VisionProcessor(age_months=AGE_MONTHS)
    
    audio_processor = None # Tạm thời tắt audio để tập trung vào các giác quan khác

    try:
        baby_model = BabyAttentionModel(
            age_months=AGE_MONTHS, hormone_system=hormone_system,
            memory_system=memory_system, vision_processor=vision_processor,
            audio_processor=audio_processor, config=config, physio_center=physio_center
        )
    except Exception as e:
        logger.critical(f"Lỗi nghiêm trọng khi khởi tạo BabyAttentionModel: {e}", exc_info=True)
        return

    # --- 2. THIẾT LẬP CAMERA VÀ VÒNG LẶP CHÍNH ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Không thể mở camera.")
        return

    # Lấy kích thước frame để khởi tạo Visualizer
    ret, frame = cap.read()
    if not ret:
        logger.error("Không thể đọc frame từ camera.")
        cap.release()
        return
    visualizer = Visualizer(frame.shape[1])

    logger.info("Camera đã được bật. Bắt đầu vòng lặp chính. Nhấn 'q' để thoát.")
    
    frame_count = 0
    # Tăng thời gian mỗi kịch bản để thấy rõ sự thay đổi tính cách
    scenario_manager = ScenarioManager(cycle_length=600)

    while True:
        ret, frame = cap.read()
        if not ret: continue

        frame_count += 1
        if frame_count % 3 != 0: continue
        
        scenario_manager.update(frame_count)
        
        mock_data = scenario_manager.get_current_sensory_input()
        sensory_input = {
            'raw_frame': frame, 'mic': {}, 'temperature_C': 37.0,
            'cognitive': {'conflict_level': random.uniform(0.0, 0.2)},
            **mock_data
        }
        
        try:
            # === CHẠY MỘT BƯỚC MÔ PHỎNG VÀ LẤY TẤT CẢ DỮ LIỆU ===
            # Đảm bảo hàm step của bạn trả về delta_vector
            # Giả định thứ tự trả về là: score, status, scores, types, delta_vector, hormone_summary, ...
            # Điều chỉnh dòng sau cho khớp với hàm step() của bạn
            results = baby_model.step(sensory_input)
            
            # Giải nén kết quả một cách an toàn
            # Giả định: score, status, scores, types, delta_vector, hormone_summary, physiological_state
            # Bạn cần kiểm tra lại hàm step() trong tuong.py để đảm bảo thứ tự này là chính xác
            # sau các lần sửa lỗi trước đó.
            score, status, _, _, delta_vector, hormone_summary, physiological_state = results

            # Chuẩn bị dữ liệu cho visualizer
            personality_profile = baby_model.emotion_integrator.get_personality_profile()
            current_ampha = baby_model.current_ampha
            
            sim_data_for_display = {
                'scenario_name': scenario_manager.get_current_scenario_name(),
                'status': status, 'score': score,
                'delta_vector': delta_vector,
                'hormone_summary': hormone_summary,
                'ampha_key': current_ampha['key'] if current_ampha else 'None',
                'ampha_intensity': current_ampha['intensity'] if current_ampha else 0,
                'personality': personality_profile,
                'physiological_state': physiological_state
            }
            
            # === HIỂN THỊ KẾT QUẢ BẰNG VISUALIZER MỚI ===
            visualizer.display(frame, sim_data_for_display)

        except Exception as e:
            logger.error(f"Lỗi trong vòng lặp chính tại bước `step`: {e}", exc_info=True)
            cv2.putText(frame, "LOI XU LY", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Mo Phong AI - Delta & Ampha', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 6. DỌN DẸP ---
    cap.release()
    cv2.destroyAllWindows()
    logger.info("--- MÔ PHỎNG ĐÃ KẾT THÚC ---")

if __name__ == '__main__':
    main()