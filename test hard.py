# test_scenario.py
import time
import numpy as np
import logging
from collections import defaultdict
import cv2
# Giả lập các import từ dự án của bạn
# Trong thực tế, bạn sẽ import trực tiếp từ các file của mình
from tuong import BabyAttentionModel, SensorProcessor
from attendcore.corehocmon2 import HormoneDynamics
from memory.expAttend import ExperienceEncoder
from sac.Vision import VisionProcessor
from sac.Audio import AudioProcessor  # Giả định bạn có file này
from attendcore.corephysiological import PhysiologicalCenter
from tho.thoemotion import EmotionIntegrationModule
from tho.thohoocmon import EmotionHarmonyBridge
from thuc import ThucConsciousnessModule, RewardEvaluator
from attendcore.coredelta import DeltaGenerator
from attendcore.corehocmon import (
    SAFETY_THRESHOLDS, SENSOR_WEIGHTS, SMELL_COMPOUNDS, 
    TASTE_COMPOUNDS, SENSITIVE_AREAS, ATTENTION_HORMONE_MAP, 
    ATTENTION_STATUS_HORMONES
)

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TestScenario')

# --- Hàm trợ giúp để tạo UI Output ---
def display_ui_output(step_name, results):
    score, status, _, types, _, hormone_summary, physiological_state = results
    
    print("\n" + "="*50)
    print(f"🎬 KỊCH BẢN: {step_name}")
    print("="*50)

    # 1. Tổng quan
    print("\n--- 📊 TỔNG QUAN CHÚ Ý ---")
    print(f"Trạng thái: {status.upper()} (Điểm: {score:.2f})")
    print(f"Loại chú ý: {', '.join(types) if types else 'Không có'}")

    # 2. Sinh lý
    print("\n--- ❤️ TRẠNG THÁI SINH LÝ ---")
    print(f"Nhịp tim: {physiological_state['heart_rate']:.0f} bpm | Nhịp thở: {physiological_state['breathing_rate']:.0f} breaths/min")
    print(f"Run rẩy: {physiological_state['tremor_level']:.2f} | Đổ mồ hôi: {physiological_state['sweating']:.2f}")

    # 3. Hormone
    print("\n--- 🧪 HỆ THỐNG HORMONE ---")
    hormone_str = " | ".join([f"{k}: {v:.2f}" for k, v in hormone_summary.items()])
    print(hormone_str)

    # 4. Ampha (Cảm xúc)
    if baby_model.current_ampha:
        ampha = baby_model.current_ampha
        print("\n--- 😊 CẢM XÚC (AMPHA) ---")
        print(f"Ampha trội: {ampha['key']} (Cường độ: {ampha['intensity']:.2f})")

    # 5. Tầng Thức
    # (Phần này sẽ cần bạn trích xuất `chosen_action` từ bên trong `step` hoặc trả về nó)
    # print("\n--- 🧠 TẦNG THỨC ---")
    # print(f"Hành động được chọn: ...")
    
    print("="*50 + "\n")


# --- Khởi tạo hệ thống ---
if __name__ == "__main__":
    AGE_MONTHS = 6
    
    # Tạo các đối tượng giả lập (mocks/stubs) nếu cần
    # Ở đây ta khởi tạo các lớp thật
    config = {
        'SAFETY_THRESHOLDS': SAFETY_THRESHOLDS,
        'SENSOR_WEIGHTS': SENSOR_WEIGHTS,
        'SMELL_COMPOUNDS': SMELL_COMPOUNDS,
        'TASTE_COMPOUNDS': TASTE_COMPOUNDS,
        'SENSITIVE_AREAS': SENSITIVE_AREAS,
        'ATTENTION_HORMONE_MAP': ATTENTION_HORMONE_MAP,
        'ATTENTION_STATUS_HORMONES': ATTENTION_STATUS_HORMONES,
        'STATUS_IMPACT_FACTOR': 1.2
    }

    # Giả lập frame ảnh đen
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Khởi tạo các thành phần
    hormone_system = HormoneDynamics()
    memory_system = ExperienceEncoder()
    vision_processor = VisionProcessor(age_months=AGE_MONTHS)
    
    # Giả lập AudioProcessor nếu không có file
    try:
        from sac.Audio import AudioProcessor
        audio_processor = AudioProcessor(age_months=AGE_MONTHS)
    except ImportError:
        logger.warning("AudioProcessor not found. Using a mock object.")
        class MockAudioProcessor:
            def get_audio_features(self): return None
            def get_contextual_features(self): return None
        audio_processor = MockAudioProcessor()

    physio_center = PhysiologicalCenter()

    # Khởi tạo mô hình chính
    baby_model = BabyAttentionModel(
        age_months=AGE_MONTHS,
        hormone_system=hormone_system,
        memory_system=memory_system,
        vision_processor=vision_processor,
        audio_processor=audio_processor,
        config=config,
        physio_center=physio_center
    )
    
    # --- Định nghĩa các kịch bản test ---

    # Kịch bản 1: Môi trường bình thường, hơi nhàm chán
    step_1_input = {
        'raw_frame': mock_frame,
        'camera': {'brightness': 0.4, 'contrast': 0.4, 'motion': 0.1, 'face_detected': False},
        'mic': {'volume_dB': 40, 'complexity': 0.2, 'is_voice': False},
        'smell': [],
        'taste': [],
        'force': [],
        'temperature_C': 37.0
    }

    # Kịch bản 2: Tiếng động lớn bất ngờ (TIÊU CỰC)
    step_2_input = step_1_input.copy()
    step_2_input['mic'] = {'volume_dB': 95, 'complexity': 0.9, 'dominant_frequency_Hz': 3000, 'is_voice': False}
    step_2_input['camera'] = {'brightness': 0.4, 'contrast': 0.4, 'motion': 0.8, 'face_detected': False}

    # Kịch bản 3: Bị kích thích xúc giác khó chịu (TIÊU CỰC)
    step_3_input = step_1_input.copy()
    step_3_input['force'] = [{'force_N': 3.0, 'body_part': 'stomach'}]

    # Kịch bản 4: Người chăm sóc xuất hiện và dỗ dành (CHUYỂN SANG TÍCH CỰC)
    # Giả lập frame có khuôn mặt
    face_frame = mock_frame.copy()
    cv2.rectangle(face_frame, (200, 100), (440, 340), (255, 255, 255), 2) # Vẽ hình chữ nhật giả khuôn mặt
    
    step_4_input = {
        'raw_frame': face_frame,
        'mic': {'volume_dB': 60, 'complexity': 0.4, 'is_voice': True, 'voice_probability': 0.9, 'sound_type': 'lullaby', 'dominant_frequency_Hz': 400},
        'smell': [{'compound': 'nonanal', 'concentration_ppm': 0.3}], # Mùi hương quen thuộc
        'taste': [],
        'force': [{'force_N': 1.0, 'body_part': 'cheek'}], # Chạm nhẹ vào má
        'temperature_C': 37.2
    }
    
    # --- Chạy các kịch bản ---
    scenarios = {
        "1. Trạng thái nghỉ, nhàm chán": step_1_input,
        "2. GIẬT MÌNH vì tiếng động lớn": step_2_input,
        "3. KHÓ CHỊU vì bị ấn vào bụng": step_3_input,
        "4. ĐƯỢC DỖ DÀNH bởi người chăm sóc": step_4_input
    }

    for name, input_data in scenarios.items():
        # Cập nhật vision features từ raw_frame nếu có
        if 'raw_frame' in input_data:
            vision_features = vision_processor.process_frame(input_data['raw_frame'])
            input_data['camera'] = vision_features

        results = baby_model.step(input_data)
        display_ui_output(name, results)
        time.sleep(1) # Nghỉ 1 giây để dễ đọc log