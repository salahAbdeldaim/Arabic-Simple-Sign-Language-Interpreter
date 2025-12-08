import os
import cv2
import time
import numpy as np

# ---------------- إعدادات المستخدم ----------------
DATA_DIR = './data_32_letters'  # مجلد حفظ البيانات (سيحتوي على 32 مجلد)
dataset_size = 80               # عدد الصور لكل فئة (قابلة للتعديل)
delay_between_saves = 0.2       # ثواني بين كل حفظ والتالي
resize_to = (224, 224)          # حجم الصورة المحفوظة
mirror_preview = False          # False => العرض معدول (non-mirrored) افتراضياً
bottom_text_height = 120        # ارتفاع المساحة السفلية لعرض النصوص
mask_preview_size = (120, 120)  # حجم الماسك المصغّر في المساحة السفلية
camera_index = 0                # رقم الكاميرا
min_contour_area = 2000         # حد بدائي لمساحة الكونتور لقبول الحفظ
# ---------------------------------------------------

# قائمة الـ 32 فئة (transliteration, arabic_label)
labels = [
    ("alif", "ا"),
    ("baa", "ب"),
    ("taa", "ت"),
    ("thaa", "ث"),
    ("jeem", "ج"),
    ("haa", "ح"),
    ("khaa", "خ"),
    ("dal", "د"),
    ("dhal", "ذ"),
    ("raa", "ر"),
    ("zay", "ز"),
    ("seen", "س"),
    ("sheen", "ش"),
    ("saad", "ص"),
    ("daad", "ض"),
    ("taa_h", "ط"),
    ("zaa_h", "ظ"),
    ("ayn", "ع"),
    ("ghayn", "غ"),
    ("faa", "ف"),
    ("qaf", "ق"),
    ("kaf", "ك"),
    ("lam", "ل"),
    ("meem", "م"),
    ("noon", "ن"),
    ("haa_h", "ه"),
    ("waw", "و"),
    ("yaa", "ي"),
    ("hamza_on_alif", "أ"),
    ("hamza_on_yaa", "ئ"),
    ("taa_marbuta", "ة"),
    ("laam_alif", "لا"),
]

# إنشاء مجلدات لكل فئة إن لم تكن موجودة
for translit, _ in labels:
    os.makedirs(os.path.join(DATA_DIR, translit), exist_ok=True)

# افتح الكاميرا
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

def detect_hand_mask(frame):
    """
    كشف يد بسيط بالاعتماد على YCrCb skin thresholds.
    ترجع (mask, max_contour_area).
    يُنصح لاحقًا باستبدالها بـ MediaPipe Hands إذا أردت دقة أعلى.
    """
    img_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower = np.array((0, 133, 77))
    upper = np.array((255, 173, 127))
    mask = cv2.inRange(img_ycrcb, lower, upper)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max([cv2.contourArea(c) for c in contours], default=0)
    return mask, max_area

def add_text_space(frame, height=bottom_text_height):
    """
    يضيف قناة سوداء أسفل الفريم لعرض النصوص والماسك المصغّر.
    """
    h, w = frame.shape[:2]
    canvas = np.zeros((h + height, w, 3), dtype=np.uint8)
    canvas[0:h, :] = frame
    return canvas

print("Controls: Q start | N skip | M toggle mirror | S skip mid-letter | ESC exit")

try:
    # حلقة على كل فئة
    for translit, arabic in labels:
        print(f"\nPreparing folder: {translit}  ({arabic})")
        folder = os.path.join(DATA_DIR, translit)

        # احسب نقطة البداية للترقيم حتى لا نكتب فوق
        existing = [f for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        start_idx = len(existing)
        print(f"Starting index for {translit}: {start_idx}  (existing files: {len(existing)})")

        # شاشة READY: عرض اسم الفئة وانتظار بدء المستخدم
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame if not mirror_preview else cv2.flip(frame, 1)
            disp = add_text_space(display_frame)
            base = display_frame.shape[0]

            cv2.putText(disp, f"Collect: {translit}   -   {arabic}", (20, base+38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(disp, "Press 'q' to start | 'n' skip | 'm' toggle mirror", (20, base+78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

            cv2.imshow("Ready", disp)
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                break
            if key == ord('n'):
                break
            if key == ord('m'):
                mirror_preview = not mirror_preview
                print("Mirror preview:", mirror_preview)
                time.sleep(0.15)

        # بدء التقاط الصور
        counter = 0
        last_time = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                break

            # طبق اختيار المرآة
            proc_frame = frame if not mirror_preview else cv2.flip(frame, 1)
            disp = add_text_space(proc_frame)
            base = proc_frame.shape[0]

            # كشف اليد (على الإطار الكامل)
            mask, max_area = detect_hand_mask(proc_frame)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # عرض النصوص في المساحة السفلية
            cv2.putText(disp, f"{translit} - {arabic}  ({start_idx + counter})", (20, base+38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(disp, f"Hand area: {int(max_area)}", (20, base+78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1, cv2.LINE_AA)
            cv2.putText(disp, "ESC: exit  |  s: skip letter  |  m: toggle mirror", (proc_frame.shape[1]-620, base+78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

            # لصق الماسك المصغّر في المساحة السفلية (يمين)
            small_mask = cv2.resize(mask_bgr, mask_preview_size)
            h0, w0 = proc_frame.shape[:2]
            disp[h0:h0+mask_preview_size[1], w0-mask_preview_size[0]:w0] = small_mask

            # شرط الحفظ: مساحة كونتور كافية + مرّ زمن التأخير
            now = time.time()
            if max_area > min_contour_area and (now - last_time) >= delay_between_saves:
                save_img = cv2.resize(proc_frame, resize_to)
                filename = f"{start_idx + counter}.jpg"   # ترقيم رقمي مرتب
                cv2.imwrite(os.path.join(folder, filename), save_img)
                counter += 1
                last_time = now

            cv2.imshow("Capture", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC للخروج الكامل
                raise KeyboardInterrupt
            if key == ord('s'):  # تخطي الحرف الحالي مبكراً
                print("Skipping this letter early.")
                break
            if key == ord('m'):
                mirror_preview = not mirror_preview
                print("Mirror preview:", mirror_preview)
                time.sleep(0.15)

        print(f"Finished {translit}. Saved {counter} new images (from {start_idx} to {start_idx+counter-1}).")

    print("\nAll done. Releasing camera.")
finally:
    cap.release()
    cv2.destroyAllWindows()
