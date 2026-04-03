import time
import numpy as np
import random
import logging
from collections import defaultdict

# --- Import tất cả các lớp và hằng số từ các file của bạn ---

# Từ sac.py và corehocmon.py
from sac.sac import SENSOR_WEIGHTS, SAFETY_THRESHOLDS, SMELL_COMPOUNDS, TASTE_COMPOUNDS, SENSITIVE_AREAS
from attendcore.corehocmon import ATTENTION_HORMONE_MAP, ATTENTION_STATUS_HORMONES

# Từ các module xử lý
from sac.Vision import VisionProcessor
from sac.Audio import AudioProcessor
from attendcore.corehocmon2 import HormoneDynamics
from memory.expAttend import ExperienceEncoder, PreferenceMemorySystem
from attendcore.corephysiological import PhysiologicalCenter
from tho.thoemotion import EmotionIntegrationModule
from tho.thohoocmon import EmotionHarmonyBridge
from tuong import BabyAttentionModel, SensorProcessor
import cv2


# --- Thiết lập Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MainSimulation')


def generate_mock_sensory_input():
    """
    Tạo dữ liệu cảm biến giả để đưa vào mô hình.
    Trong thực tế, dữ liệu này sẽ đến từ camera, micro, v.v.
    """
    # Dữ liệu thị giác giả
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(mock_frame, f"Frame time: {time.time():.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Dữ liệu cảm biến khác
    sensory_input = {
        'raw_frame': mock_frame,
        'camera': { # Dữ liệu này sẽ được ghi đè bởi VisionProcessor
            'flicker_freq': random.uniform(0, 40),
            'clarity': random.uniform(0.3, 0.9),
            'contrast': random.uniform(0.4, 0.8),
            'saturation': random.uniform(0.5, 1.0),
            'hue_R': random.uniform(0.1, 0.9),
            'hue_G': random.uniform(0.1, 0.5),
            'hue_B': random.uniform(0.1, 0.5),
            'motion': random.uniform(0.0, 1.0),
            'face_detected': random.choice([True, False]),
            'object_size': random.uniform(0.05, 0.4),
            'object_distance': random.uniform(0.5, 1.5)
        },
        'mic': { # Dữ liệu này sẽ được ghi đè bởi AudioProcessor
            'volume_dB': random.uniform(40, 85),
            'dominant_frequency_Hz': random.uniform(100, 4000),
            'is_voice': random.choice([True, False]),
            'is_rhythmic': random.choice([True, False]),
            'sound_type': 'unknown',
            'voice_probability': random.uniform(0.5, 0.9)
        },
        'smell': [
            {'compound': random.choice(['nonanal', 'hexanoic_acid', 'limonene']), 
             'concentration_ppm': random.uniform(0.05, 0.8)}
        ],
        'taste': [
            {'compound': random.choice(['lactose', 'quinine', 'citric_acid']), 
             'concentration_mg_dl': random.uniform(5, 20)}
        ],
        'force': [
            {'body_part': 'hand', 'force_N': random.uniform(0.5, 2.5), 'is_tickle': random.choice([False, True])}
        ],
        'temperature_C': random.uniform(36.0, 38.0),
        'cognitive': {
            'conflict_level': random.uniform(0.1, 0.8)
        }
    }
    return sensory_input

def print_status(step, score, status, types, hormone_summary, physio_state, current_ampha):
    """In trạng thái hiện tại của mô phỏng một cách gọn gàng."""
    print("\n" + "="*80)
    print(f"--- STEP {step} ---")
    print(f"  Attention Score: {score:.3f} | Status: {status}")
    print(f"  Attention Types: {types if types else 'None'}")
    
    print("\n  Hormone Levels:")
    hormones_str = " | ".join([f"{k}: {v:.2f}" for k, v in hormone_summary.items()])
    print(f"    {hormones_str}")

    print("\n  Physiological State:")
    physio_str = " | ".join([f"{k}: {v:.2f}" for k, v in physio_state.items()])
    print(f"    {physio_str}")

    if current_ampha:
        print("\n  Dominant Ampha (Emotion):")
        ampha_str = f"Key: {current_ampha['key']}, Intensity: {current_ampha['intensity']:.2f}, Components: {current_ampha['components']}"
        print(f"    {ampha_str}")
    else:
        print("\n  Dominant Ampha (Emotion): None")
    print("="*80)


def main():
    """Hàm chính để chạy mô phỏng."""
    AGE_MONTHS = 6
    SIMULATION_STEPS = 20

    logger.info(f"Starting simulation for a {AGE_MONTHS}-month-old infant.")

    # --- 1. Tải cấu hình ---
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

    # --- 2. Khởi tạo các module ---
    logger.info("Initializing system components...")
    hormone_system = HormoneDynamics()
    memory_system = ExperienceEncoder()
    vision_processor = VisionProcessor(age_months=AGE_MONTHS)
    audio_processor = AudioProcessor(age_months=AGE_MONTHS)
    physio_center = PhysiologicalCenter()

    # Khởi tạo mô hình chú ý chính và "tiêm" các thành phần phụ thuộc
    model = BabyAttentionModel(
        age_months=AGE_MONTHS,
        hormone_system=hormone_system,
        memory_system=memory_system,
        vision_processor=vision_processor,
        audio_processor=audio_processor,
        config=config,
        physio_center=physio_center
    )
    
    # SỬA LỖI: Thêm các thuộc tính còn thiếu vào instance của model
    # (Cách tốt nhất là thêm vào file tuong.py)
    model.amphas = model.emotion_integrator.amphas
    model.personality_traits = defaultdict(float)


    # --- 3. Chạy vòng lặp mô phỏng ---
    try:
        model.start_audio_processing()
        logger.info("--- Simulation Started ---")

        for i in range(SIMULATION_STEPS):
            # Tạo dữ liệu cảm biến mới cho mỗi bước
            sensory_input = generate_mock_sensory_input()

            # Cập nhật một số điều kiện y tế ngẫu nhiên
            model.update_medical_conditions(is_hungry=random.choice([True, False]))

            # Thực hiện một bước tính toán của mô hình
            score, status, scores, types, _, hormone_summary, physio_state = model.step(sensory_input)

            # In kết quả
            print_status(i + 1, score, status, types, hormone_summary, physio_state, model.current_ampha)

            # Đợi một chút trước khi sang bước tiếp theo
            time.sleep(2)

    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")
    except Exception as e:
        logger.error(f"An error occurred during simulation: {e}", exc_info=True)
    finally:
        # Dọn dẹp tài nguyên
        model.stop_audio_processing()
        logger.info("--- Simulation Ended ---")


if __name__ == "__main__":
    main()