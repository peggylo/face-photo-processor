import cv2
import os
from PIL import Image
import numpy as np

def process_images():
    # 取得當前目錄中的所有圖片 測試
    image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("找不到圖片檔案")
        return
    
    # 建立輸出資料夾 備註
    output_dir = 'processed_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img_file in image_files:
        # 使用OpenCV讀取圖片進行人臉偵測
        img_cv = cv2.imread(img_file)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 載入人臉檢測器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # 取得人臉位置
            x, y, w, h = faces[0]
            
            # 計算上半身區域（擴大人臉區域）
            center_x = x + w//2
            center_y = y + h//2
            
            # 決定正方形邊長（使用較大的值以確保包含上半身）
            square_size = int(max(w, h) * 2.5)  # 可以調整這個倍數
            
            # 計算正方形的邊界
            left = max(0, center_x - square_size//2)
            top = max(0, center_y - square_size//2)
            right = min(img_cv.shape[1], left + square_size)
            bottom = min(img_cv.shape[0], top + square_size)
            
            # 裁切圖片
            cropped = img_cv[top:bottom, left:right]
            
            # 轉換回PIL格式並保存
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            
            # 確保輸出為正方形
            size = min(cropped_pil.size)
            output_size = (size, size)
            
            # 調整大小並保存
            cropped_pil = cropped_pil.resize(output_size, Image.Resampling.LANCZOS)
            output_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + '_processed.png')
            cropped_pil.save(output_path, 'PNG')
            print(f"已處理: {img_file}")
        else:
            print(f"在 {img_file} 中未偵測到人臉")

if __name__ == "__main__":
    process_images()
