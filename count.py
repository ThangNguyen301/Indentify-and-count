from ultralytics import YOLO
import cv2
import os

# ================= MODEL =================
model = YOLO("runs/detect/train2/weights/best.pt")

# ================= CHỌN NGUỒN (chỉ bật 1 dòng) =================
#source = r"E:\Identify and count\images\hq720 (1).jpg"      # ẢNH
#source = r"E:\Identify and count\videos\18394353-hd_1080_1920_30fps.mp4"  # VIDEO
source = 0                                                              # WEBCAM

# ================= CẤU HÌNH CHUNG =================
line_y = 340                    # ← Điều chỉnh theo video (300-450)
counted_ids = set()
total_count = 0
frame_count = 0

# ================= FUNCTION XỬ LÝ =================
def process_frame(frame, is_image=False):
    global total_count, counted_ids, frame_count
    frame_count += 1

    # Resize giữ tỷ lệ
    h, w = frame.shape[:2]
    scale = 640 / max(h, w)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # ================= TRACKING TỐI ƯU =================
    results = model.track(
        frame,
        persist=True,
        conf=0.40,
        iou=0.45,
        tracker="botsort.yaml",      # ổn định nhất cho video
        imgsz=640,
        augment=is_image,
        max_det=50
    )

    for r in results:
        if r.boxes is None or r.boxes.id is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, obj_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Vẽ box + ID + conf
            color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {int(obj_id)} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # ================= LOGIC ĐẾM =================
            if is_image:
                # Ảnh tĩnh: đếm tất cả cá detect được
                if obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    total_count += 1
            else:
                # Video/Webcam: đếm khi cá vượt đường line từ trên xuống
                if cy > line_y and obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    total_count += 1

    # Vẽ đường line (chỉ video/webcam)
    if not is_image:
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 3)

    # Hiển thị thông tin
    cv2.putText(frame, f"Total: {total_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    return frame


# ================= CHẠY THEO NGUỒN =================
is_image_mode = isinstance(source, str) and os.path.isfile(source) and source.lower().endswith((".jpg", ".png", ".jpeg"))

if is_image_mode:
    # ================= ẢNH TĨNH =================
    frame = cv2.imread(source)
    if frame is None:
        print("❌ Không đọc được ảnh!")
        exit()

    frame = process_frame(frame, is_image=True)
    cv2.imshow("YOLO Image Tracking", frame)
    print(f"✅ Xong ảnh - Tổng: {total_count} con")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # ================= VIDEO / WEBCAM =================
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    print("Nhấn ESC để thoát | Nhấn R để reset đếm")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, is_image=False)

        cv2.imshow("YOLO Fish Counting", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:          # ESC
            break
        elif key == ord('r') or key == ord('R'):
            counted_ids.clear()
            total_count = 0
            print("🔄 Đã reset bộ đếm!")

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Kết thúc - Tổng cá đếm được: {total_count}")