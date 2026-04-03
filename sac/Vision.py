import cv2
import numpy as np
import math
from collections import deque, defaultdict
import logging
# --- Thiết lập Logging ---
logger = logging.getLogger(__name__)
class VisionProcessor:
    def __init__(self, age_months):
        self.age_months = age_months
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.prev_frame = None
        self.motion_history = deque(maxlen=10)
        self.flicker_history = deque(maxlen=5)
        self.flicker_freq = 0
        self.min_contour_area = 500
        self.object_tracks = defaultdict(lambda: {
            'positions': deque(maxlen=15),
            'features': deque(maxlen=15),
            'last_seen': 0,
            'object_id': 0
        })
        self.next_object_id = 1
        self.frame_count = 0
        self.spatial_relations = {}
        self.edge_detection_params = {
            'blur_size': 5,
            'canny_low_ratio': 0.5,
            'canny_high_ratio': 1.0
        }
        self.tracking_params = {
            'max_missed_frames': 5,
            'similarity_threshold': 0.7
        }
        logger.info("VisionProcessor initialized.")
        self.proto_history = {}
        self.proto_stability_threshold= 5
        self.frame_count = 0

    def process_frame(self, frame):
        """Xử lý frame ảnh và trích xuất các đặc trưng thị giác nâng cao"""
        # Chuyển đổi sang ảnh xám để xử lý
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Tính toán các đặc trưng cơ bản
        features = {
            'brightness': self._calculate_brightness(gray),
            'contrast': self._calculate_contrast(gray),
            'motion': self._detect_motion(gray),
            'face_detected': self._detect_faces(gray),
            'hue_R': 0, 'hue_G': 0, 'hue_B': 0,
            'saturation': 0,
            'flicker_freq': self._update_flicker_detection(gray),
            'object_size': 0,
            'object_distance': 1.0,
            'geometric_features': [],
            'spatial_relations': {},
            'structural_features': [],
            'tracked_objects': [],
            'proto_objects': []
        }
        
        # Phân tích màu sắc

        # Phát hiện kích thước vật thể nếu có khuôn mặt
        if features['face_detected']:
            logger.debug("Face detected in the frame.")
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                features['object_size'] = w * h / (frame.shape[0] * frame.shape[1])
                features['object_distance'] = max(0.3, min(2.0, 1.0 - (w / frame.shape[1]) * 1.7))
                
        
        # Trích xuất đặc trưng hình học
        edges = self._detect_edges(gray)
        contours = self._find_contours(edges)
        geometric_features = self._analyze_contours(contours, frame.shape[:2])
        features['geometric_features'] = geometric_features
        color_features = self._analyze_color(frame, geometric_features)
        features.update(color_features)
        logger.debug(f"Vision features: Objects={len(geometric_features)}, "
                    f"Tracked={len(features['tracked_objects'])}, "
                    f"R={features['hue_R']:.2f}, G={features['hue_G']:.2f}, "
                    f"Motion={features['motion']:.2f}")                
        
        # Phân tích mối quan hệ không gian giữa các contour
        if geometric_features:
            spatial_analysis = self._analyze_spatial_relations(geometric_features)
            features['spatial_relations'] = spatial_analysis
            
            # Phân tích cấu trúc chi tiết
            structural_features = self._analyze_structural_details(edges, geometric_features)
            features['structural_features'] = structural_features
            
            # Theo dõi đối tượng qua thời gian
            tracked_objects = self._track_objects(geometric_features)
            features['tracked_objects'] = tracked_objects
            if tracked_objects:
                logger.debug(f"Tracking {len(tracked_objects)} objects.")
            # Tạo proto-objects (nhóm các contour có quan hệ)
            proto_objects = self._form_proto_objects(geometric_features, spatial_analysis)
            features['proto_objects'] = proto_objects
      
        self.frame_count += 1
        logger.debug(f"Finished processing frame. Motion: {features['motion']:.3f}, Faces: {features['face_detected']}")
        return {
            'brightness': features['brightness'],
            'contrast': features['contrast'],
            'motion': features['motion'],
            'face_detected': features['face_detected'],
            'hue_R': features['hue_R'],
            'hue_G': features['hue_G'],
            'hue_B': features['hue_B'],
            'saturation': features['saturation'],
            'flicker_freq': features['flicker_freq'],
            'object_size': features['object_size'],
            'object_distance': features['object_distance'],
            'geometric_features': features['geometric_features'],
            'spatial_relations': features['spatial_relations'],
            'structural_features': features['structural_features'],
            'tracked_objects': [
                {
                    'id': obj['object_id'], #
                    'positions': obj['trajectory'], #
                    'shape_type': obj['shape_type'], #
                    'stability': obj['stability'], #
                    'familiar': obj['familiar'],   # Bổ sung
                    'reappeared': obj['reappeared'] # Bổ sung
                } for obj in features['tracked_objects']
            ],
            'proto_objects': [
                {
                    'components': proto['components'], #
                    'bounding_box': proto['bounding_box'], #
                    'complexity': proto['complexity'], #
                    'stability': proto['stability'], #
                    'type': proto['type'] #
                } for proto in features['proto_objects']
            ]
        }
    # Thêm vào cuối hàm process_frame

    # ======== CÁC PHƯƠNG THỨC CƠ BẢN ========
    
    def _calculate_brightness(self, gray_frame):
        """Tính độ sáng trung bình của ảnh"""
        return np.mean(gray_frame) / 255.0

    def _calculate_contrast(self, gray_frame):
        """Tính độ tương phản bằng độ lệch chuẩn"""
        return np.std(gray_frame) / 255.0

    def _detect_motion(self, gray_frame):
        """Phát hiện chuyển động bằng phương pháp so sánh với frame trước"""
        if self.prev_frame is None:
            self.prev_frame = gray_frame
            return 0.0
        
        # Tính toán sự khác biệt giữa các frame
        frame_diff = cv2.absdiff(self.prev_frame, gray_frame)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Tính tỷ lệ pixel thay đổi
        motion_level = np.sum(thresh) / (255 * thresh.size)
        self.prev_frame = gray_frame
        
        # Cập nhật lịch sử chuyển động
        self.motion_history.append(motion_level)
        
        # Tính trung bình chuyển động trong lịch sử
        if len(self.motion_history) > 0:
            return sum(self.motion_history) / len(self.motion_history)
        return 0.0

    def _update_flicker_detection(self, gray_frame):
        """Cập nhật phát hiện nhấp nháy dựa trên lịch sử độ sáng"""
        brightness = np.mean(gray_frame)
        self.flicker_history.append(brightness)
        
        if len(self.flicker_history) < 5:
            return 0.0
        
        # Tính toán sự thay đổi độ sáng giữa các frame liên tiếp
        diffs = []
        for i in range(1, len(self.flicker_history)):
            diffs.append(abs(self.flicker_history[i] - self.flicker_history[i-1]))
        
        avg_diff = sum(diffs) / len(diffs)
        
        # Nếu có sự thay đổi đáng kể, cập nhật tần số nhấp nháy
        if avg_diff > 20:  # Ngưỡng thay đổi độ sáng
            self.flicker_freq = min(120, self.flicker_freq + 5)
        else:
            self.flicker_freq = max(0, self.flicker_freq - 2)
        
        return self.flicker_freq

    def _detect_faces(self, gray_frame):
        """Phát hiện khuôn mặt trong frame"""
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.1, 4)
        return len(faces) > 0

    def _analyze_color(self, color_frame, geometric_features):
        """Phân tích màu sắc tập trung vào các vật thể chính"""
        # Khởi tạo giá trị mặc định
        color_features = {
            'hue_R': 0, 'hue_G': 0, 'hue_B': 0,
            'saturation': 0
        }

        if not geometric_features:
            return color_features

        # Tạo mask tập trung vào các vật thể
        mask = np.zeros(color_frame.shape[:2], dtype=np.uint8)
        for feat in geometric_features:
            x, y, w, h = [int(v) for v in feat['bounding_box_unnormalized']]
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

        # Phân tích màu trong khu vực vật thể
        hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        mean_saturation = cv2.mean(hsv[:,:,1], mask=mask)[0] / 255.0

        # Tính toán màu chủ đạo
        b, g, r = cv2.split(color_frame)
        r_mean = cv2.mean(r, mask=mask)[0] / 255.0
        g_mean = cv2.mean(g, mask=mask)[0] / 255.0
        b_mean = cv2.mean(b, mask=mask)[0] / 255.0

        return {
            'hue_R': r_mean,
            'hue_G': g_mean,
            'hue_B': b_mean,
            'saturation': mean_saturation
        }

    # ======== PHÂN TÍCH HÌNH HỌC NÂNG CAO ========
    
    def _detect_edges(self, gray_frame):
        """
        Phát hiện các đường biên (cạnh) trong ảnh xám
        Sử dụng thuật toán Canny với ngưỡng tự động
        """
        # Làm mờ ảnh để giảm nhiễu
        blur_size = self.edge_detection_params['blur_size']
        blurred = cv2.GaussianBlur(gray_frame, (blur_size, blur_size), 0)
        
        # Tính toán ngưỡng tự động sử dụng phương pháp Otsu
        high_thresh, thresh_im = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        low_thresh = self.edge_detection_params['canny_low_ratio'] * high_thresh
        high_thresh = self.edge_detection_params['canny_high_ratio'] * high_thresh
        
        # Áp dụng phát hiện cạnh Canny
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        return edges

    def _find_contours(self, edge_map):
        """
        Tìm các đường bao (contour) từ bản đồ cạnh
        Trả về danh sách các contour hợp lệ
        """
        # Tìm contours với chế độ RETR_EXTERNAL (chỉ lấy contour ngoài cùng)
        contours, _ = cv2.findContours(
            edge_map, 
            mode=cv2.RETR_EXTERNAL, 
            method=cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def _analyze_contours(self, contours, frame_shape):
        """
        Phân tích các contour để trích xuất thông tin hình học có ý nghĩa
        Trả về danh sách các đặc trưng hình học đã được chuẩn hóa
        """
        img_height, img_width = frame_shape
        geometric_features = []
        
        for contour in contours:
            # Bỏ qua contour quá nhỏ (nhiễu)
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # Tính toán các thuộc tính hình học cơ bản
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Tính moment để tìm trọng tâm
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + w/2, y + h/2
            
            # Tính độ tròn (circularity)
            if perimeter > 0:
                circularity = (4 * math.pi * area) / (perimeter ** 2)
            else:
                circularity = 0
            
            # Tính tỷ lệ khung hình (aspect ratio)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Xấp xỉ contour bằng đa giác
            epsilon = 0.02 * perimeter  # Tham số độ chính xác
            approx = cv2.approxPolyDP(contour, epsilon, True)
            num_vertices = len(approx)
            
            # Chuẩn hóa các giá trị theo kích thước frame
            normalized_area = area / (img_width * img_height)
            normalized_perimeter = perimeter / (img_width + img_height)
            normalized_center = (cx / img_width, cy / img_height)
            normalized_bbox = (
                x / img_width, 
                y / img_height, 
                w / img_width, 
                h / img_height
            )
            
            # Phân loại hình dạng cơ bản dựa trên số đỉnh
            if num_vertices == 3:
                shape_type = "triangle"
            elif num_vertices == 4:
                # Kiểm tra xem có phải hình vuông không
                aspect_ratio = w / float(h)
                if 0.95 <= aspect_ratio <= 1.05:
                    shape_type = "square"
                else:
                    shape_type = "rectangle"
            elif num_vertices > 4 and circularity > 0.85:
                shape_type = "circle"
            elif num_vertices > 4:
                shape_type = "polygon"
            else:
                shape_type = "irregular"
            
            # Lưu trữ dữ liệu contour gốc để phân tích không gian
            contour_points = contour.squeeze().tolist() if contour.shape[0] > 2 else []
            
            # Xác định contour có phải là đường thẳng không
            is_line = False
            orientation = 0
            if num_vertices == 2:
                is_line = True
                # Tính hướng của đường thẳng
                dx = approx[1][0][0] - approx[0][0][0]
                dy = approx[1][0][1] - approx[0][0][1]
                orientation = math.degrees(math.atan2(dy, dx)) % 360
            
            # Tạo đối tượng đặc trưng hình học
            feature = {
                "area": normalized_area,
                "perimeter": normalized_perimeter,
                "center": normalized_center,
                "bounding_box": normalized_bbox,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "num_vertices": num_vertices,
                "shape_type": shape_type,
                "contour_points": contour_points,
                "center_unnormalized": (cx, cy),
                "bounding_box_unnormalized": (x, y, w, h),
                "frame_width": img_width,
                "frame_height": img_height,
                "is_line": is_line,
                "orientation": orientation
            }
            
            geometric_features.append(feature)
        
        return geometric_features

    # ======== PHÂN TÍCH KHÔNG GIAN NÂNG CAO ========
    
    def _analyze_spatial_relations(self, features):
        """
        Phân tích mối quan hệ không gian giữa các contour
        Trả về từ điển với các mối quan hệ không gian
        """
        relations = {}
        
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i == j:
                    continue
                    
                # Kiểm tra mối quan hệ bao bọc
                contains = self._check_containment(feat1, feat2)
                if contains:
                    relations.setdefault(i, {})[j] = {'relation': 'contains'}
                    continue
                
                # Kiểm tra khoảng cách và hướng
                distance, angle = self._calculate_spatial_relationship(feat1, feat2)
                relations.setdefault(i, {})[j] = {
                    'relation': 'adjacent' if distance < 0.1 else 'distant',
                    'distance': distance,
                    'angle': angle
                }
                
                # Kiểm tra song song (cho các contour dạng đường thẳng)
                if feat1.get('is_line', False) and feat2.get('is_line', False):
                    angle_diff = abs(feat1['orientation'] - feat2['orientation'])
                    if angle_diff < 10 or angle_diff > 170:  # Gần song song
                        relations[i][j]['relation'] = 'parallel'
        
        return relations
    
    def _check_containment(self, feat1, feat2):
        """
        Kiểm tra xem contour1 có chứa contour2 không
        """
        # Kiểm tra nếu contour1 không đủ điểm để tạo polygon
        if len(feat1['contour_points']) < 3:
            return False
            
        # Sử dụng kiểm tra điểm trong đa giác
        test_point = (int(feat2['center_unnormalized'][0]), int(feat2['center_unnormalized'][1]))
        contour_points = np.array(feat1['contour_points'], dtype=np.int32)
        
        result = cv2.pointPolygonTest(contour_points, test_point, False)
        return result >= 0  # >=0: nằm trên hoặc bên trong contour
    
    def _calculate_spatial_relationship(self, feat1, feat2):
        """
        Tính khoảng cách và góc giữa hai contour
        """
        x1, y1 = feat1['center_unnormalized']
        x2, y2 = feat2['center_unnormalized']
        
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx**2 + dy**2)
        
        # Chuẩn hóa khoảng cách theo đường chéo ảnh
        max_distance = math.sqrt(feat1['frame_width']**2 + feat1['frame_height']**2)
        normalized_distance = distance / max_distance if max_distance > 0 else 0
        
        # Tính góc (độ) so với trục ngang
        angle = math.degrees(math.atan2(dy, dx)) % 360 if dx != 0 or dy != 0 else 0
        
        return normalized_distance, angle

    # ======== PHÂN TÍCH CẤU TRÚC CHI TIẾT ========
    
    def _analyze_structural_details(self, edge_map, features):
        """
        Trích xuất các đặc trưng cấu trúc chi tiết:
        - Độ cong tại các điểm
        - Điểm nút giao nhau
        - Texture cục bộ
        """
        structural_features = []
        
        # Tìm điểm giao nhau của các cạnh
        junctions = self._find_junction_points(edge_map)
        
        for i, feat in enumerate(features):
            # Tính toán độ cong tại các điểm
            curvature = self._calculate_curvature(feat['contour_points'])
            
            # Kiểm tra điểm nút gần nhất
            closest_junction = None
            min_dist = float('inf')
            center_x, center_y = feat['center_unnormalized']
            
            for jx, jy in junctions:
                dist = math.sqrt((jx - center_x)**2 + (jy - center_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_junction = (jx, jy)
            
            # Trích xuất texture cục bộ (sử dụng LBP đơn giản)
            texture = self._extract_local_texture(edge_map, feat['bounding_box_unnormalized'])
            
            structural_features.append({
                'id': i,
                'curvature': curvature,
                'nearest_junction': closest_junction,
                'texture': texture
            })
        
        return structural_features
    
    def _find_junction_points(self, edge_map):
        """
        Tìm các điểm nút giao nhau của các cạnh
        """
        # Tạo kernel để phát hiện giao điểm
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
        
        # Áp dụng convolution
        neighbor_count = cv2.filter2D(edge_map, -1, kernel)
        
        # Tìm điểm có >= 3 lân cận là cạnh
        junctions = []
        ys, xs = np.where(neighbor_count >= 3)
        for x, y in zip(xs, ys):
            junctions.append((x, y))
        
        return junctions
    
    def _calculate_curvature(self, contour_points):
        """
        Tính độ cong tại các điểm dọc theo contour
        """
        if len(contour_points) < 3:
            return []
            
        curvatures = []
        points = np.array(contour_points)
        
        for i in range(1, len(points)-1):
            p0 = points[i-1]
            p1 = points[i]
            p2 = points[i+1]
            
            # Tính vector tiếp tuyến
            v1 = p1 - p0
            v2 = p2 - p1
            
            # Tính góc giữa các vector
            dot_product = np.dot(v1, v2)
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                curvature = 1.0 / (angle + 1e-5)  # Tránh chia cho 0
                curvatures.append(curvature)
        
        return curvatures
    
    def _extract_local_texture(self, edge_map, bbox):
        """
        Trích xuất texture cục bộ trong bounding box
        """
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Đảm bảo ROI nằm trong kích thước ảnh
        h, w_img = edge_map.shape
        x = max(0, min(x, w_img-1))
        y = max(0, min(y, h-1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h - y))
        
        roi = edge_map[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 0
        
        # Tính toán mật độ cạnh (đơn giản)
        edge_density = np.sum(roi) / (255 * roi.size)
        return edge_density

    # ======== THEO DÕI ĐỐI TƯỢNG (OBJECT PERMANENCE) ========
    
    def _track_objects(self, current_features):
        """
        Theo dõi các đối tượng qua các khung hình
        """
        tracked_objects = []
        current_time = self.frame_count
        
        # Tạo bản đồ tính năng hiện tại để so khớp
        current_feature_map = {i: feat for i, feat in enumerate(current_features)}
        
        # Cập nhật các đối tượng hiện có
        for obj_id, track_data in list(self.object_tracks.items()):
            reappeared_flag = False
            # Kiểm tra nếu đối tượng đã biến mất tạm thời và giờ xuất hiện lại
            if current_time - track_data['last_seen'] > 1 and \
               current_time - track_data['last_seen'] <= self.tracking_params['max_missed_frames']:
                reappeared_flag = True
                logger.debug(f"Object {obj_id} reappeared.") #

            if current_time - track_data['last_seen'] > self.tracking_params['max_missed_frames']:
                del self.object_tracks[obj_id]
                continue

            best_match = None
            best_score = -1

            # Tìm tính năng tốt nhất phù hợp trong khung hình hiện tại
            for feat_id, feat in current_feature_map.items():
                similarity = self._feature_similarity(
                    track_data['features'][-1],  # Tính năng cuối cùng được theo dõi
                    feat
                )

                if similarity > best_score:
                    best_score = similarity
                    best_match = feat_id

            # Nếu tìm thấy kết quả phù hợp tốt
            threshold = self.tracking_params['similarity_threshold']
            if best_match is not None and best_score > threshold:
                matched_feat = current_feature_map.pop(best_match)
                track_data['positions'].append(matched_feat['center_unnormalized'])
                track_data['features'].append(matched_feat)
                track_data['last_seen'] = current_time

                # Tính toán độ ổn định (số frame đã theo dõi)
                stability = min(1.0, len(track_data['positions']) / 15.0) #

                # Xác định 'familiar'
                # Nếu đối tượng đã được theo dõi trong một số frame tối thiểu (ví dụ: 5 frame)
                is_familiar = len(track_data['positions']) >= 5 #

                tracked_objects.append({
                    'object_id': track_data['object_id'],
                    'current_position': matched_feat['center'],
                    'trajectory': list(track_data['positions']),
                    'stability': stability,
                    'shape_type': matched_feat['shape_type'],
                    'familiar': is_familiar,   #
                    'reappeared': reappeared_flag #
                })

        # Tạo đối tượng mới cho các tính năng không khớp
        for feat_id, feat in current_feature_map.items():
            new_obj_id = self.next_object_id
            self.next_object_id += 1

            self.object_tracks[new_obj_id] = {
                'positions': deque([feat['center_unnormalized']], maxlen=15),
                'features': deque([feat], maxlen=15),
                'last_seen': current_time,
                'object_id': new_obj_id
            }

            tracked_objects.append({
                'object_id': new_obj_id,
                'current_position': feat['center'],
                'trajectory': [feat['center_unnormalized']],
                'stability': 0.0, # Đối tượng mới bắt đầu, độ ổn định ban đầu là 0
                'shape_type': feat['shape_type'],
                'familiar': False,   # Đối tượng mới chưa quen thuộc
                'reappeared': False  # Đối tượng mới không phải tái xuất hiện
            })

        return tracked_objects
    
    def _feature_similarity(self, feat1, feat2):
        """
        Tính điểm tương đồng giữa hai tính năng
        """
        # Khoảng cách giữa các tâm
        dist = math.sqrt((feat1['center_unnormalized'][0] - feat2['center_unnormalized'][0])**2 +
                         (feat1['center_unnormalized'][1] - feat2['center_unnormalized'][1])**2)
        
        # Sự khác biệt về diện tích
        area_diff = abs(feat1['area'] - feat2['area'])
        
        # Sự khác biệt về hình dạng
        shape_diff = 0
        if feat1['shape_type'] == feat2['shape_type']:
            shape_diff = 0
        elif feat1['shape_type'] in ['circle', 'square'] and feat2['shape_type'] in ['circle', 'square']:
            shape_diff = 0.3
        else:
            shape_diff = 0.7
        
        # Tính điểm tổng hợp (càng cao càng giống)
        similarity = 1.0 - min(1.0, dist/100.0 + area_diff + shape_diff)
        return similarity

    # ======== TẠO PROTO-OBJECTS ========
    
    def _form_proto_objects(self, features, spatial_relations):
        """
        Nhóm các contour có quan hệ không gian thành proto-objects
        """
        proto_objects = []
        grouped = set()
        
        for i, feat in enumerate(features):
            if i in grouped:
                continue
                
            # Tìm tất cả các contour có liên quan
            related = self._find_related_features(i, spatial_relations)
            grouped.update(related)
            
            if len(related) > 1:
                # Tạo proto-object từ nhóm các contour
                proto_feats = [features[idx] for idx in related]
                combined_bbox = self._combine_bboxes(proto_feats)
                
                # Tính độ phức tạp dựa trên số contour và hình dạng
                complexity = len(related) * 0.3
                for feat in proto_feats:
                    if feat['shape_type'] in ['polygon', 'irregular']:
                        complexity += 0.2
                # Xác định loại proto-object
                obj_type = 'unknown'
                if any('contains' in spatial_relations.get(idx, {}) for idx in related):
                    obj_type = 'container'
                elif all(f['shape_type'] == 'circle' for f in proto_feats):
                    obj_type = 'circular_group'
                
                # Tính độ ổn định (dựa trên sự tồn tại qua các frame)
                stability = 0.0
                proto_key = tuple(sorted(related))
                if proto_key in self.proto_history:
                    stability = min(1.0, self.proto_history[proto_key]['count'] / 10.0)
                    self.proto_history[proto_key]['count'] += 1
                    self.proto_history[proto_key]['last_seen'] = self.frame_count
                else:
                    self.proto_history[proto_key] = {
                        'count': 1,
                        'last_seen': self.frame_count,
                        'first_seen': self.frame_count
                    }
                
                # Loại bỏ proto-object cũ
                for key in list(self.proto_history.keys()):
                    if self.frame_count - self.proto_history[key]['last_seen'] > 30:
                        del self.proto_history[key]
                
                proto_objects.append({
                    'components': list(related),
                    'bounding_box': combined_bbox,
                    'complexity': min(3.0, complexity),
                    'stability': stability,
                    'type': obj_type
                })
    
        return proto_objects
    
    def _find_related_features(self, start_idx, relations, visited=None):
        """
        Tìm tất cả các contour có liên quan thông qua DFS
        """
        if visited is None:
            visited = set()
            
        visited.add(start_idx)
        related = [start_idx]
        
        if start_idx in relations:
            for neighbor, rel_data in relations[start_idx].items():
                if neighbor not in visited and rel_data['relation'] in ['contains', 'adjacent']:
                    related.extend(self._find_related_features(neighbor, relations, visited))
        
        return related
    
    def _combine_bboxes(self, features):
        """
        Kết hợp các bounding box thành một bbox chung
        """
        min_x = min(feat['bounding_box_unnormalized'][0] for feat in features)
        min_y = min(feat['bounding_box_unnormalized'][1] for feat in features)
        max_x = max(feat['bounding_box_unnormalized'][0] + feat['bounding_box_unnormalized'][2] for feat in features)
        max_y = max(feat['bounding_box_unnormalized'][1] + feat['bounding_box_unnormalized'][3] for feat in features)
        
        width = max_x - min_x
        height = max_y - min_y
        
        normalized_bbox = (
            min_x / features[0]['frame_width'],
            min_y / features[0]['frame_height'],
            width / features[0]['frame_width'],
            height / features[0]['frame_height']
        )
        
        return normalized_bbox