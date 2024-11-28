import cv2
import numpy as np
from PIL import Image
import os
import mediapipe as mp
import ssl

# 忽略 SSL 證書驗證
ssl._create_default_https_context = ssl._create_unverified_context

def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    # 創建抗鋸齒的遮罩
    mask = np.zeros((h, w), dtype=np.float32)
    mask[dist_from_center <= radius] = 1.0
    
    # 添加更寬的平滑過渡區域
    transition_width = 2.0
    mask[dist_from_center <= radius + transition_width] = np.clip(1.0 - (dist_from_center[dist_from_center <= radius + transition_width] - radius) / transition_width, 0, 1)
    
    # 應用高斯模糊來實現更平滑的邊緣
    mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
    
    return (mask * 255).astype(np.uint8)

def process_image(image_path, output_path):
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        print(f"無法讀取圖片: {image_path}")
        return
    
    # 初始化 MediaPipe
    mp_pose = mp.solutions.pose
    
    # 轉換顏色空間
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # 使用 MediaPipe 偵測人體
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5) as pose:
        
        results = pose.process(rgb_img)
        
        if results.pose_landmarks:
            # 獲取上半身的關鍵點
            landmarks = results.pose_landmarks.landmark
            
            # 獲取肩膀和臉部的關鍵點
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            
            # 計算上半身的範圍
            center_x = int((left_shoulder.x + right_shoulder.x) * w / 2)
            center_y = int((nose.y + (left_shoulder.y + right_shoulder.y) / 2) * h / 2)
            
            # 計算裁切範圍
            shoulder_width = abs(right_shoulder.x - left_shoulder.x) * w
            head_to_shoulder = abs(nose.y - (left_shoulder.y + right_shoulder.y) / 2) * h
            
            # 使用較大的值來確保圓形區域足夠大
            crop_size = int(max(shoulder_width * 3.0, head_to_shoulder * 4.0))
            
            # 確保裁切範圍不會太小
            crop_size = max(crop_size, int(min(w, h) * 0.4))
            
            # 計算裁切區域，保持長寬比
            x1 = max(0, center_x - crop_size // 2)
            y1 = max(0, center_y - crop_size // 2)
            x2 = min(w, x1 + crop_size)
            y2 = min(h, y1 + crop_size)
            
            # 調整裁切區域確保正方形
            crop_width = x2 - x1
            crop_height = y2 - y1
            if crop_width > crop_height:
                diff = crop_width - crop_height
                y1 = max(0, y1 - diff // 2)
                y2 = min(h, y2 + diff // 2)
            elif crop_height > crop_width:
                diff = crop_height - crop_width
                x1 = max(0, x1 - diff // 2)
                x2 = min(w, x2 + diff // 2)
            
            # 裁切圖片
            cropped = img[y1:y2, x1:x2]
            
            if cropped.size == 0:
                print(f"裁切後圖片為空: {image_path}")
                return
            
            # 保持原始解析度
            crop_h, crop_w = cropped.shape[:2]
            size = max(crop_h, crop_w)
            
            # 創建一個帶有 alpha 通道的正方形背景
            square_img = np.zeros((size, size, 4), dtype=np.uint8)
            
            # 將裁切後的圖片放在正方形背景的中央
            y_offset = (size - crop_h) // 2
            x_offset = (size - crop_w) // 2
            
            # 將 BGR 圖片轉換為 BGRA
            cropped_bgra = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
            
            # 設置原始圖片區域為完全不透明
            cropped_bgra[:, :, 3] = 255
            
            # 將圖片放入正方形背景
            square_img[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = cropped_bgra
            
            # 創建圓形遮罩（使用原始的二值遮罩）
            Y, X = np.ogrid[:size, :size]
            center = (size//2, size//2)
            radius = size//2 - 2  # 稍微縮小半徑以避免邊緣鋸齒
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
            
            # 創建平滑的遮罩
            mask = np.zeros((size, size), dtype=np.float32)
            
            # 內圈完全不透明
            inner_radius = radius - 2
            mask[dist_from_center <= inner_radius] = 1.0
            
            # 外圈漸變透明
            transition_width = 3
            outer_mask = np.clip(1.0 - (dist_from_center - inner_radius) / transition_width, 0, 1)
            mask = np.maximum(mask, outer_mask)
            
            # 應用遮罩到 alpha 通道
            square_img[:, :, 3] = (mask * 255).astype(np.uint8)
            
            # 使用 PIL 保存為 PNG
            pil_img = Image.fromarray(cv2.cvtColor(square_img, cv2.COLOR_BGRA2RGBA))
            pil_img.save(output_path, "PNG")
            print(f"成功處理並保存: {output_path}")
        else:
            print(f"在圖片中沒有檢測到人: {image_path}")

def main():
    # 建立輸出資料夾
    output_dir = "cropped_speakers"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 支援的圖片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.webp')
    
    # 處理當前資料夾中的所有圖片
    for filename in os.listdir('.'):
        if filename.lower().endswith(supported_formats):
            output_path = os.path.join(output_dir, f"cropped_{os.path.splitext(filename)[0]}.png")
            print(f"正在處理: {filename}")
            try:
                process_image(filename, output_path)
            except Exception as e:
                print(f"處理 {filename} 時發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()
