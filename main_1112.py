import cv2
import os
from PIL import Image
import numpy as np
from PIL import ImageDraw
from io import BytesIO
import shutil

def compress_image(image, max_size_mb=1):
    """壓縮圖片至指定大小（MB）以下"""
    quality = 95
    while True:
        temp_buffer = BytesIO()
        image.save(temp_buffer, format='PNG', quality=quality)
        size_mb = temp_buffer.tell() / (1024 * 1024)
        
        if size_mb <= max_size_mb or quality <= 5:
            return image, quality
        
        quality -= 5

def process_images():
    try:
        # 載入人臉檢測器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 取得圖片檔案列表
        image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print("找不到圖片檔案")
            return
            
        print(f"找到 {len(image_files)} 個圖片檔案")
        
        # 建立輸出資料夾
        circle_dir = 'circle_images'
        compressed_dir = 'compressed_circle_images'
        for directory in [circle_dir, compressed_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        for img_file in image_files:
            try:
                print(f"處理圖片: {img_file}")
                # 使用 PIL 讀取圖片（解決中文檔名問題）
                pil_image = Image.open(img_file)
                img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                if img_cv is None:
                    print(f"無法讀取圖片: {img_file}")
                    continue
                    
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                # 人臉檢測
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4
                )
                
                if len(faces) > 0:
                    print(f"在 {img_file} 中檢測到 {len(faces)} 個人臉")
                    x, y, w, h = faces[0]
                    
                    # 計算中心點
                    center_x = x + w//2
                    center_y = y + h//2
                    
                    # 決定正方形邊長
                    square_size = int(max(w, h) * 2.5)
                    
                    # 計算裁切區域
                    left = max(0, center_x - square_size//2)
                    top = max(0, center_y - square_size//2)
                    right = min(img_cv.shape[1], left + square_size)
                    bottom = min(img_cv.shape[0], top + square_size)
                    
                    # 確保裁切區域不會超出圖片範圍
                    width = right - left
                    height = bottom - top
                    size = min(width, height)
                    
                    # 重新調整裁切區域
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
                    face_img = img_cv[int(top):int(bottom), int(left):int(right)]
                    
                    # 轉換回PIL格式
                    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    
                    # 建立圓形遮罩
                    mask = Image.new('L', face_pil.size, 0)
                    draw = ImageDraw.Draw(mask)
                    draw.ellipse((0, 0, face_pil.size[0]-1, face_pil.size[1]-1), fill=255)
                    
                    # 套用圓形遮罩
                    output = Image.new('RGBA', face_pil.size, (0, 0, 0, 0))
                    output.paste(face_pil, (0, 0))
                    output.putalpha(mask)
                    
                    # 儲存原始圓形版本
                    circle_path = os.path.join(circle_dir, f'circle_{os.path.splitext(img_file)[0]}.png')
                    output.save(circle_path, 'PNG')
                    
                    # 檢查檔案大小並處理壓縮版本
                    file_size_mb = os.path.getsize(circle_path) / (1024 * 1024)
                    
                    if file_size_mb > 1:
                        print(f"檔案大小為 {file_size_mb:.2f}MB，進行壓縮...")
                        compressed_img, used_quality = compress_image(output)
                        compressed_path = os.path.join(compressed_dir, f'compressed_circle_{os.path.splitext(img_file)[0]}.png')
                        compressed_img.save(compressed_path, 'PNG', quality=used_quality)
                        new_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                        print(f"已壓縮至 {new_size_mb:.2f}MB")
                    else:
                        # 如果檔案小於 1MB，直接複製
                        compressed_path = os.path.join(compressed_dir, f'compressed_circle_{os.path.splitext(img_file)[0]}.png')
                        shutil.copy2(circle_path, compressed_path)
                        print(f"檔案大小為 {file_size_mb:.2f}MB，直接複製")
                    
                    print(f"已儲存圓形版本：\n- {circle_path}\n- {compressed_path}")
                else:
                    print(f"在 {img_file} 中未檢測到人臉")
                    
            except Exception as e:
                print(f"處理圖片 {img_file} 時發生錯誤: {str(e)}")
                continue
                
    except Exception as e:
        print(f"程式執行發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_images()
