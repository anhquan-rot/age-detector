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
    
    # Kiểm tra xem có file ảnh được gửi lên không
    if 'image' not in request.files:
        return jsonify({"error": "Không tìm thấy file ảnh trong request"}), 400

    file = request.files['image']
    
    try:
        # Đọc file ảnh từ request
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Không thể đọc file ảnh"}), 400

        # --- Sử dụng DeepFace để phân tích ---
        # `enforce_detection=False` để không báo lỗi nếu không tìm thấy khuôn mặt
        # `actions` chỉ định chúng ta muốn phân tích những gì
        logging.info("Bắt đầu phân tích bằng DeepFace...")
        analysis_results = DeepFace.analyze(
            img_path=img, 
            actions=['age'], 
            enforce_detection=False
        )
        logging.info("Phân tích hoàn tất.")

        # `analysis_results` là một danh sách, mỗi phần tử là một khuôn mặt
        results_to_send = []
        for face_data in analysis_results:
            # `face_data` là một dictionary chứa kết quả cho một khuôn mặt
            age = face_data.get('age')
            region = face_data.get('region') # Tọa độ [x, y, w, h]
            
            warning = ""
            if age and age < 13:
                warning = "CẢNH BÁO: Phát hiện trẻ em dưới 13 tuổi!"
            
            # Chuyển đổi box từ [x, y, w, h] sang [x1, y1, x2, y2]
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