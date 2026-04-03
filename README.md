# YukiAI: Mô hình Mô phỏng Tâm sinh lý và Ý thức Trẻ sơ sinh

**YukiAI** (trước đây là BabyBuda) là dự án nền tảng được phát triển trước khi hình thành hệ sinh thái **Penta**. Đây là một hệ thống mô phỏng phức tạp về các phản ứng tâm sinh xử lý, động lực học hormone và các tầng ý thức của trẻ sơ sinh dựa trên các mô hình toán học và AI tiên tiến.

---

## 🌟 Tổng quan dự án

YukiAI không chỉ là một chatbot thông thường. Nó là một nỗ lực nhằm tái hiện cách một bộ não sơ khai phản ứng với thế giới thông qua:
1.  **Hệ thống Hormone**: Mô phỏng sự biến đổi của Dopamine, Oxytocin, Cortisol, Serotonin... theo thời gian thực.
2.  **Xử lý Đa giác quan**: Tích hợp thị giác (Vision) và thính giác (Audio) để đánh giá mức độ chú ý.
3.  **Hệ thống Ý thức (Thức)**: Sử dụng mạng LSTM để mô phỏng các xu hướng tâm lý và ý định hành động.
4.  **Cầu nối sLLM**: Tương tác thông minh qua các mô hình ngôn ngữ lớn (như Qwen) chạy offline qua Ollama.

## 🏗️ Cấu trúc thư mục

```text
📁 yukiai/
├── 📁 attendcore/       # Nhân cốt lõi về tâm sinh lý và hormone
│   ├── coredelta.py     # Tính toán sự thay đổi trạng thái
│   ├── corehocmon.py    # Định nghĩa bản đồ hormone và cảm giác
│   └── corephysiological.py # Trung tâm điều khiển sinh lý
├── 📁 sac/              # Cảm biến và Xử lý tín hiệu (SAC - Sensory Action Control)
│   ├── Audio.py         # Phân tích đặc trưng âm thanh
│   ├── Vision.py        # Xử lý hình ảnh và nhận diện
│   └── sac.py           # Cấu hình trọng số cảm biến theo tháng tuổi
├── 📁 memory/           # Hệ thống lưu trữ và mã hóa trải nghiệm
├── 📁 tho/              # Tích hợp cảm xúc và hòa hợp nội tại
├── 📁 tho/              # Tầng Ý thức và Đánh giá phần thưởng (Reward Evaluator)
├── tuong.py             # Script tích hợp chính (Main Engine)
├── sllm_bridge.py       # Cầu nối với Ollama / sLLM
└── README.md            # Tài liệu dự án
```

## 🚀 Các tính năng chính

### 1. Mô phỏng Hormone Động (Hormone Dynamics)
Hệ thống theo dõi các mức hormone như:
- **Dopamine**: Động lực và sự tò mò.
- **Oxytocin**: Sự gắn kết và cảm giác an toàn.
- **Cortisol**: Căng thẳng và phản ứng với kích thích tiêu cực.
- **GABA**: Sự ức chế và điều hòa thần kinh.

### 2. Đánh giá Chú ý theo tháng tuổi
Mô hình tự điều chỉnh trọng số các giác quan dựa trên tuổi của trẻ (0-24 tháng). Ví dụ: trẻ sơ sinh chú trọng vào thính giác và khứu giác hơn thị giác.

### 3. Tầng Ý thức (Consciousness Layer)
Sử dụng kiến trúc LSTM để duy trì "trạng thái tâm trí" xuyên suốt, giúp YukiAI không chỉ phản ứng tức thời mà còn có các xu hướng hành động như: Khám phá, Tránh né, Tìm sự an ủi, hoặc Nghỉ ngơi.

### 4. Tích hợp sLLM Offline
Sử dụng Ollama để chạy các mô hình AI mạnh mẽ một cách riêng tư, cho phép YukiAI "suy nghĩ" và giao tiếp dựa trên trạng thái cảm xúc hiện tại.

## 🛠️ Yêu cầu hệ thống

- **Python**: 3.8+
- **Thư viện chính**: `numpy`, `torch`, `ollama`, `opencv-python`.
- **Phần cứng**: Khuyên dùng GPU để xử lý Vision và sLLM mượt mà.

## 📖 Hướng dẫn khởi chạy

1.  **Cài đặt Ollama**: Tải và cài đặt tại [ollama.com](https://ollama.com).
2.  **Tải mô hình**: `ollama pull qwen2.5:7b` (hoặc mô hình bạn chọn).
3.  **Cài đặt thư viện**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Chạy ứng dụng**:
    ```bash
    python tuong.py
    ```

---

## 📅 Lịch sử phát triển
YukiAI là viên gạch đầu tiên, đặt tiền đề cho các nghiên cứu sâu hơn về AI tương tác và hệ sinh thái **Penta**. Dự án tập trung vào việc tạo ra một thực thể số có khả năng "cảm nhận" trước khi "tư duy".

---
*Phát triển bởi USER & YukiAI Team.*
