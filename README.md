# Face Photo Processor

自動化頭像處理工具，專門用於處理演講者照片。這個工具可以自動檢測人像、裁切上半身，並創建帶有平滑邊緣的圓形透明背景照片。

本專案使用 [Windsurf](https://www.codeium.com/windsurf) 和 Cascade AI 協助開發，展示了如何利用現代 AI 工具來加速開發流程並提升程式碼品質。

## 功能特點

- 自動檢測人像並裁切上半身
- 創建圓形遮罩，帶有平滑的邊緣過渡
- 輸出透明背景的 PNG 圖片
- 支援多種圖片格式 (jpg, jpeg, png, webp)
- 保持原始圖片品質
- 批次處理多張圖片

## 系統需求

- Python 3.12 或更高版本
- 需要網路連接（首次執行時下載 MediaPipe 模型）

## 安裝

1. 克隆此專案：
```bash
git clone https://github.com/peggylo/face-photo-processor.git
cd face-photo-processor
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 將要處理的圖片放在程式所在的資料夾中
2. 執行程式：
```bash
python crop_speaker_photos.py
```
3. 處理完的圖片會儲存在 `cropped_speakers` 資料夾中

## 注意事項

- 建議使用正面、清晰的人像照片
- 照片中應該要能清楚看到人的上半身
- 輸出的檔案命名格式為：`cropped_[原始檔名].png`

## 開發工具

- 使用 Windsurf IDE 和 Cascade AI 協助開發
- 透過 AI 輔助優化程式碼結構和效能
- 運用 AI 協作提升開發效率和程式碼品質
