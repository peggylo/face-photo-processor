import cv2
import os
from PIL import Image
import numpy as np
from PIL import ImageDraw

def process_images():
    # 取得當前目錄中的所有圖片
    image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("找不到圖片檔案")
        return
    
    # 建立兩個輸出資料夾
    square_dir = 'square_images'
    circle_dir = 'circle_images'
    for directory in [square_dir, circle_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    for img_file in image_files:
        # 使用OpenCV讀取圖片進行人臉偵測
        img_cv = cv2.imread(img_file)
        if img_cv is None:
            print(f"無法讀取圖片: {img_file}")
            continue
            
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 1. 調整圖片大小到合適的尺寸
        max_size = 1024
        scale = max_size / max(gray.shape)
        if scale < 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
            
        # 2. 增加對比度
        gray = cv2.equalizeHist(gray)
        
        # 3. 使用高斯模糊減少噪點
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 4. 調整人臉檢測參數
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # 改回較小的值以提高檢測率
            minNeighbors=3,   # 降低閾值
            minSize=(100, 100)  # 增加最小檢測尺寸，因為人臉在圖中比較大
        )
        
        # 載入人臉檢測器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # 修改 scaleFactor 和 minNeighbors 參數來提高檢測穩定性
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,  # 從1.1改為1.3，減少檢測次數但提高穩定性
            minNeighbors=5,   # 從4改為5，減少誤檢
            minSize=(30, 30)  # 添加最小檢測尺寸
        )
        
        # 添加日誌來幫助診斷
        if len(faces) > 0:
            print(f"檢測到人臉位置: {faces[0]}")
        else:
            print(f"未檢測到人臉，圖片尺寸: {gray.shape}")
        
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
            
            # 確保裁切區域不會超出圖片範圍
            width = right - left
            height = bottom - top
            size = min(width, height)
            
            # 重新調整裁切區域使其成為正方形
            left = center_x - size//2
            right = left + size
            top = center_y - size//2
            bottom = top + size
            
            # 確保邊界不會超出圖片範圍
            if left < 0:
                left = 0
                right = size
            if right > img_cv.shape[1]:
                right = img_cv.shape[1]
                left = right - size
            if top < 0:
                top = 0
                bottom = size
            if bottom > img_cv.shape[0]:
                bottom = img_cv.shape[0]
                top = bottom - size
                
            # 裁切圖片
            cropped = img_cv[int(top):int(bottom), int(left):int(right)]
            
            # 轉換回PIL格式並保存
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            
            # 確保輸出為正方形
            size = min(cropped_pil.size)
            output_size = (size, size)
            
            # 調整大小並保存正方形版本
            cropped_pil = cropped_pil.resize(output_size, Image.Resampling.LANCZOS)
            square_path = os.path.join(square_dir, os.path.splitext(img_file)[0] + '_square.png')
            cropped_pil.save(square_path, 'PNG')

            # 創建圓形遮罩
            mask = Image.new('L', output_size, 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.ellipse((0, 0, size-1, size-1), fill=255)

            # 創建圓形版本
            output = Image.new('RGBA', output_size, (0, 0, 0, 0))
            output.paste(cropped_pil, (0, 0))
            output.putalpha(mask)
            
            # 保存圓形版本
            circle_path = os.path.join(circle_dir, os.path.splitext(img_file)[0] + '_circle.png')
            output.save(circle_path, 'PNG')
            
            print(f"已處理: {img_file}")
        else:
            print(f"在 {img_file} 中未偵測到人臉")

if __name__ == "__main__":
    process_images()
