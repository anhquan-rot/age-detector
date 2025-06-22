from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import cv2
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# --- KHÔNG CẦN TẢI MODEL TRƯỚC ---
# DeepFace sẽ tự động tải model khi cần thiết

# --- Định nghĩa route để upload và xử lý ảnh ---
@app.route('/detect_age', methods=['POST'])
def detect_age():
    logging.info("Nhận được request...")
    
    image_data = None
    
    # Ưu tiên đọc file từ form 'multipart/form-data'
    if 'image' in request.files:
        logging.info("Tìm thấy file trong request.files (form-data).")
        file = request.files['image']
        image_data = file.read()
    # Nếu không có, thử đọc dữ liệu thô (raw binary)
    elif request.data:
        logging.info("Không có file trong form-data, thử đọc request.data (raw binary).")
        image_data = request.data
    else:
        return jsonify({"error": "Không tìm thấy dữ liệu ảnh trong request"}), 400

    try:
        # Đọc file ảnh từ dữ liệu đã lấy được
        npimg = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Không thể đọc dữ liệu ảnh (invalid format)"}), 400

        # --- Phần còn lại của hàm xử lý giữ nguyên ---
        logging.info("Bắt đầu phân tích bằng DeepFace...")
        analysis_results = DeepFace.analyze(
            img_path=img, 
            actions=['age'], 
            enforce_detection=False
        )
        # ... (toàn bộ phần xử lý và trả về JSON giữ nguyên như trước)
        # ...
        logging.info("Phân tích hoàn tất.")
        results_to_send = []
        for face_data in analysis_results:
            age = face_data.get('age')
            region = face_data.get('region')
            warning = ""
            if age and age < 13:
                warning = "CẢNH BÁO: Phát hiện trẻ em dưới 13 tuổi!"
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            box = [x, y, x + w, y + h]
            results_to_send.append({
                "box": box,
                "estimated_age": age,
                "warning": warning
            })
            logging.info(f"Phát hiện khuôn mặt, tuổi dự đoán: {age}. Cảnh báo: '{warning}'")
        if not results_to_send:
            return jsonify({"message": "Không tìm thấy khuôn mặt nào."}), 200
        return jsonify({"detections": results_to_send})

    except Exception as e:
        logging.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
        return jsonify({"error": "Lỗi server nội bộ."}), 500


# Route cơ bản để kiểm tra server có hoạt động không
@app.route('/')
def index():
    return "Hello, Age Detection Server with DeepFace is running!"

if __name__ == '__main__':
    # Chạy ở chế độ debug khi ở local
    app.run(host='0.0.0.0', port=5000, debug=True)