from ultralytics import YOLO
import cv2
import os
import ctypes

# ================= MODEL =================
model = YOLO("runs/detect/train2/weights/best.pt")

# ================= CHỌN NGUỒN =================
#source = r"E:\Identify and count\images\hq720 (1).jpg"      # ẢNH
#source = r"E:\Identify and count\videos\18394353-hd_1080_1920_30fps.mp4"  # VIDEO
source = 0                                                  # WEBCAM

# ================= LẤY ĐỘ PHÂN GIẢI MÀN HÌNH =================
user32 = ctypes.windll.user32
screen_w = user32.GetSystemMetrics(0)
screen_h = user32.GetSystemMetrics(1)

# ================= CẤU HÌNH =================
line_y = 340
counted_ids = set()
total_count = 0
frame_count = 0

# ================= FUNCTION =================
def process_frame(frame, is_image=False):
    global total_count, counted_ids, frame_count
    frame_count += 1

    # Resize YOLO input (giữ tỷ lệ để detect chuẩn)
    h, w = frame.shape[:2]
    scale = 640 / max(h, w)
    frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # ================= TRACKING =================
    results = model.track(
        frame_resized,
        persist=True,
        conf=0.40,
        iou=0.45,
        tracker="botsort.yaml",
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

            # scale lại về frame gốc
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Vẽ box
            color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {int(obj_id)} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # ================= ĐẾM =================
            if is_image:
                if obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    total_count += 1
            else:
                if cy > line_y and obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    total_count += 1

    # Vẽ line
    if not is_image:
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 3)

    # Hiển thị tổng
    cv2.putText(frame, f"Total: {total_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    # ================= FULL MÀN HÌNH =================
    frame = cv2.resize(frame, (screen_w, screen_h))

    return frame


# ================= CHECK ẢNH =================
is_image_mode = isinstance(source, str) and os.path.isfile(source) and source.lower().endswith((".jpg", ".png", ".jpeg"))

# ================= ẢNH =================
if is_image_mode:
    frame = cv2.imread(source)
    if frame is None:
        print("❌ Không đọc được ảnh!")
        exit()

    frame = process_frame(frame, is_image=True)

    cv2.namedWindow("YOLO", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("YOLO", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("YOLO", frame)

    print(f"✅ Xong ảnh - Tổng: {total_count}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ================= VIDEO / WEBCAM =================
else:
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    # FULLSCREEN WINDOW
    cv2.namedWindow("YOLO", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("YOLO", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("ESC: thoát | R: reset")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, is_image=False)

        cv2.imshow("YOLO", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('r') or key == ord('R'):
            counted_ids.clear()
            total_count = 0
            print("🔄 Reset!")

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Tổng : {total_count}")