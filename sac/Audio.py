import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt # Thay filtfilt bằng sosfiltfilt
from collections import deque
import threading
import sounddevice as sd
import time
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AudioProcessor')

class AudioStreamer:
    """Lớp thu âm liên tục với bộ đệm vòng hiệu quả"""
    def __init__(self, sample_rate=16000, channels=1, buffer_duration=2.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = int(sample_rate * buffer_duration)
        self.buffer = np.zeros((self.buffer_size, channels), dtype=np.float32)
        self.write_index = 0
        self.lock = threading.Lock()
        self.stream = None
        self.is_recording = False

    def start(self):
        """Bắt đầu thu âm"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._callback,
            blocksize=1024
        )
        self.stream.start()
        logger.info("Audio streaming started")

    def _callback(self, indata, frames, time, status):
        """Callback xử lý dữ liệu âm thanh"""
        if status:
            logger.warning(f"Audio stream status: {status}")
            
        with self.lock:
            available_space = self.buffer_size - self.write_index
            if frames <= available_space:
                self.buffer[self.write_index:self.write_index+frames] = indata
                self.write_index += frames
            else:
                # Xử lý tràn bộ đệm
                self.buffer[:available_space] = indata[:available_space]
                self.buffer[0:frames-available_space] = indata[available_space:]
                self.write_index = frames - available_space

    def get_recent_audio(self, duration=1.0):
        """Lấy đoạn âm thanh gần nhất với độ dài chỉ định"""
        samples_needed = int(self.sample_rate * duration)
        with self.lock:
            if samples_needed > self.buffer_size:
                samples_needed = self.buffer_size
                
            start_index = self.write_index - samples_needed
            if start_index < 0:
                # Lấy dữ liệu từ cả đầu và cuối buffer
                first_part = self.buffer[start_index:]
                second_part = self.buffer[:self.write_index]
                return np.vstack((first_part, second_part))
            else:
                return self.buffer[start_index:self.write_index]

    def stop(self):
        """Dừng thu âm"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.is_recording = False
        logger.info("Audio streaming stopped")


class AudioFeatureExtractor:
    """Trích xuất đặc trưng âm thanh hiệu quả"""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # Tạo bộ lọc thông dải cho âm thanh giọng nói
        self.sos_speech = butter(4, [80, 3000], 'bandpass', fs=sample_rate, output='sos')
        
    def extract_features(self, audio_data):
        """Trích xuất các đặc trưng âm thanh cơ bản"""
        if len(audio_data) == 0:
            return None
            
        # Chuyển đổi stereo sang mono nếu cần
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        features = {}
        
        # 1. Tính độ ồn tổng thể (dB)
        rms = np.sqrt(np.mean(audio_data**2))
        features['volume_dB'] = 20 * np.log10(rms + 1e-9)
        
        # 2. Tính toán cao độ sử dụng autocorrelation
        features['pitch_Hz'] = self._estimate_pitch(audio_data)
        
        # 3. Tính tần số trội
        features['dominant_frequency_Hz'] = self._get_dominant_frequency(audio_data)
        
        # 4. Tính độ phức tạp phổ
        features['spectral_complexity'] = self._calculate_spectral_complexity(audio_data)
        
        # 5. Kiểm tra tính nhịp điệu
        features['is_rhythmic'] = self._detect_rhythm(audio_data)
        
        # 6. Tách giọng nói
        speech_features = self._analyze_speech(audio_data)
        features.update(speech_features)
        
        return features
        
    def _estimate_pitch(self, audio_data):
        """Ước lượng cao độ sử dụng autocorrelation"""
        if len(audio_data) < 1024:
            return 0
            
        # Chuẩn hóa dữ liệu âm thanh
        audio_data = audio_data - np.mean(audio_data)
        audio_data = audio_data / np.max(np.abs(audio_data)) + 1e-9
        
        # Tính autocorrelation
        autocorr = np.correlate(audio_data, audio_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Tìm đỉnh đầu tiên sau đỉnh trung tâm
        min_index = int(self.sample_rate / 1000)  # Tần số tối thiểu 1000Hz
        max_index = int(self.sample_rate / 80)    # Tần số tối đa 80Hz
        autocorr = autocorr[min_index:max_index]
        
        if len(autocorr) == 0:
            return 0
            
        peak_index = np.argmax(autocorr) + min_index
        return self.sample_rate / peak_index if peak_index > 0 else 0
        
    def _get_dominant_frequency(self, audio_data):
        """Xác định tần số trội trong âm thanh"""
        if len(audio_data) < 512:
            return 0
            
        fft = np.abs(np.fft.rfft(audio_data))
        freqs = np.fft.rfftfreq(len(audio_data), 1.0/self.sample_rate)
        return freqs[np.argmax(fft)] if len(fft) > 0 else 0
        
    def _calculate_spectral_complexity(self, audio_data):
        """Tính độ phức tạp của phổ âm thanh"""
        if len(audio_data) < 1024:
            return 0
            
        # Tính spectrogram
        S = np.abs(librosa.stft(audio_data, n_fft=512))
        
        # Tính entropy phổ
        spectral_entropy = -np.sum(S * np.log(S + 1e-9), axis=0)
        return np.mean(spectral_entropy)
        
    def _detect_rhythm(self, audio_data):
        """Phát hiện tính nhịp điệu trong âm thanh"""
        if len(audio_data) < 2048:
            return False
            
        try:
            # Tính onset envelope
            onset_env = librosa.onset.onset_strength(
                y=audio_data, 
                sr=self.sample_rate,
                hop_length=512
            )
            
            # Phát hiện nhịp
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=self.sample_rate,
                hop_length=512
            )
            
            return len(beats) > 2  # Có ít nhất 3 nhịp
        except:
            return False
            
    def _analyze_speech(self, audio_data):
        """Phân tích đặc điểm giọng nói"""
        features = {
            'sound_type': 'unknown',
            'voice_probability': 0.0,
            'is_voice': False
        }
        
        if len(audio_data) < 1024:
            return features
            
        try:
            # Lọc lấy dải tần giọng nói
            filtered = sosfiltfilt(self.sos_speech, audio_data) # Sử dụng hàm sosfiltfilt
            # Tính các đặc trưng phổ
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=filtered, sr=self.sample_rate
            ))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(
                y=filtered, sr=self.sample_rate
            ))
            
            # Phân loại dựa trên đặc trưng
            if 100 < spectral_centroid < 1000 and spectral_bandwidth < 2000:
                features['sound_type'] = 'speech_like'
                features['voice_probability'] = 0.8
                features['is_voice'] = True
            elif spectral_centroid > 2000 and spectral_bandwidth > 3000:
                features['sound_type'] = 'noise'
                features['voice_probability'] = 0.1
            else:
                features['sound_type'] = 'other'
                features['voice_probability'] = 0.3
                
            # Trích xuất MFCC nếu là giọng nói
            if features['is_voice']:
                mfcc = librosa.feature.mfcc(
                    y=filtered, 
                    sr=self.sample_rate, 
                    n_mfcc=13
                )
                features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
                features['mfcc_std'] = np.std(mfcc, axis=1).tolist()
                
        except Exception as e:
            logger.error(f"Speech analysis error: {e}")
            
        return features


class AudioProcessor:
    """Bộ xử lý âm thanh chính tích hợp các thành phần"""
    def __init__(self, age_months, sample_rate=16000, buffer_duration=2.0):
        self.age_months = age_months
        self.sample_rate = sample_rate
        self.streamer = AudioStreamer(sample_rate, buffer_duration=buffer_duration)
        self.feature_extractor = AudioFeatureExtractor(sample_rate)
        self.last_processed_time = time.time()
        self.feature_cache = deque(maxlen=10)  # Cache 10 frame gần nhất

    def start(self):
        """Bắt đầu xử lý âm thanh"""
        self.streamer.start()
        logger.info("Audio processing started")

    def stop(self):
        """Dừng xử lý âm thanh"""
        self.streamer.stop()
        logger.info("Audio processing stopped")

    def get_audio_features(self):
        """Lấy đặc trưng âm thanh mới nhất"""
        current_time = time.time()
        
        # Lấy dữ liệu âm thanh 1 giây gần nhất
        audio_data = self.streamer.get_recent_audio(duration=1.0)
        
        # Kiểm tra dữ liệu hợp lệ
        if audio_data is None or len(audio_data) == 0:
            return None
            
        # Trích xuất đặc trưng
        features = self.feature_extractor.extract_features(audio_data)
        
        # Bổ sung thông tin thời gian
        if features:
            logger.debug(f"Audio features extracted: "
                        f"Volume={features.get('volume_dB', 0):.1f} dB, "
                         f"Pitch={features.get('pitch_Hz', 0):.1f} Hz, "
                         f"Type='{features.get('sound_type', 'N/A')}'")
            features['processing_time'] = time.time() - current_time
            features['timestamp'] = current_time
            self.feature_cache.append(features)
        
        return features

    def get_contextual_features(self, window_size=3):
        """Lấy đặc trưng có ngữ cảnh trong khoảng thời gian"""
        if len(self.feature_cache) < window_size:
            return None
            
        # Lấy các frame gần nhất
        recent_features = list(self.feature_cache)[-window_size:]
        
        # Tính toán các đặc trưng tổng hợp
        contextual = {
            'volume_trend': self._calculate_trend([f['volume_dB'] for f in recent_features]),
            'pitch_variation': np.std([f['pitch_Hz'] for f in recent_features]),
            'voice_probability_avg': np.mean([f.get('voice_probability', 0) for f in recent_features]),
            'is_continuous_speech': all(f.get('is_voice', False) for f in recent_features),
            'frame_count': len(recent_features)
        }
        logger.debug(f"Contextual features calculated: "
                     f"Volume Trend={contextual.get('volume_trend', 0)}, "
                     f"Cont. Speech={contextual.get('is_continuous_speech', False)}")
        return contextual
        
    def _calculate_trend(self, values):
        """Tính xu hướng của dãy giá trị"""
        if len(values) < 2:
            return 0
            
        # Tính hệ số góc của đường hồi quy tuyến tính
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.5:
            return 1  # Tăng
        elif slope < -0.5:
            return -1  # Giảm
        else:
            return 0  # Ổn định


# Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo bộ xử lý cho bé 12 tháng
    processor = AudioProcessor(age_months=12)
    
    try:
        processor.start()
        time.sleep(2)  # Chờ thu đủ dữ liệu
        
        # Lấy đặc trưng âm thanh
        features = processor.get_audio_features()
        if features:
            print("\nĐặc trưng âm thanh cơ bản:")
            print(f"Volume: {features['volume_dB']:.1f} dB")
            print(f"Pitch: {features['pitch_Hz']:.1f} Hz")
            print(f"Sound type: {features['sound_type']}")
            
            # Lấy đặc trưng ngữ cảnh
            contextual = processor.get_contextual_features()
            if contextual:
                print("\nĐặc trưng ngữ cảnh:")
                print(f"Volume trend: {contextual['volume_trend']}")
                print(f"Continuous speech: {contextual['is_continuous_speech']}")
                
    finally:
        processor.stop()