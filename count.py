from ultralytics import YOLO
import cv2
import os

# ================= MODEL =================
model = YOLO("runs/detect/train2/weights/best.pt")

# ================= CHỌN NGUỒN =================
source = r"E:\Identify and count\images\hq720 (1).jpg"
# source = 0                    # Webcam
# source = "video.mp4"          # Video

# ================= CẤU HÌNH =================
line_y = 320                    # Vị trí đường đếm (tùy chỉnh theo video)
counted_ids = set()
total_count = 0

# ================= FUNCTION =================
def process_frame(frame, is_image=False):
    global total_count, counted_ids

    # Resize giữ tỷ lệ tốt hơn
    h, w = frame.shape[:2]
    scale = 640 / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    frame = cv2.resize(frame, (new_w, new_h))

    # Track
    results = model.track(
        frame,
        persist=True,
        conf=0.45,           # Có thể chỉnh 0.4 ~ 0.55
        iou=0.5,
        tracker="botsort.yaml",
        imgsz=640,
        augment= is_image    # Tăng augment cho ảnh tĩnh
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

            # Vẽ box và ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {int(obj_id)} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # ================= LOGIC ĐẾM =================
            if is_image:
                # === ẢNH TĨNH: Đếm tất cả cá detect được ===
                if obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    total_count += 1
            else:
                # === VIDEO / WEBCAM: Đếm khi qua đường ===
                if cy > line_y and obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    total_count += 1

    # Vẽ đường đếm (chỉ hiện khi là video/webcam)
    if not is_image:
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)

    # Hiển thị tổng
    cv2.putText(frame, f"Total: {total_count}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return frame


# ================= CHẠY THEO NGUỒN =================
if isinstance(source, str) and os.path.isfile(source) and source.lower().endswith((".jpg", ".png", ".jpeg")):
    # ================= XỬ LÝ ẢNH =================
    frame = cv2.imread(source)
    if frame is None:
        print("❌ Không đọc được ảnh")
        exit()

    frame = process_frame(frame, is_image=True)

    cv2.imshow("YOLO Image Tracking", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # ================= VIDEO HOẶC WEBCAM =================
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Giảm lag webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, is_image=False)

        cv2.imshow("YOLO Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC để thoát
            break

    cap.release()
    cv2.destroyAllWindows()