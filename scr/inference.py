import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display

# =============================================================
# 🎛️ إعدادات الموديلات والمسارات
# =============================================================
# قمنا بتعريف الموديلات في قاموس لسهولة التبديل
MODELS_CONFIG = {
    '1': {
        'name': 'KNN (Jana)',
        'path': r"models/knn_model.pkl"
    },
    '2': {
        'name': 'Linear SVC (Abdelaziz)',
        'path': r"models/LinearSVC_model.pkl"
    },
    '3': {
        'name': 'Logistic_Reg (Tarek)',
        'path': r"models/Logistic_Reg.pkl"
    },
    '4': {
        'name': 'Random Forest (Mohamed)',
        'path': r"models/RainForcement_Model.p"
    }
}

# =============================================================
# 🛠️ دوال مساعدة (تحميل الموديل + استخراج الميزات)
# =============================================================

def safe_load_model(path):
    """تحميل الموديل سواء كان كائناً مباشراً أو داخل قاموس"""
    if not os.path.exists(path):
        print(f"❌ خطأ: الملف غير موجود: {path}")
        return None
    
    try:
        loaded_obj = joblib.load(path)
        # التحقق مما إذا كان الملف قاموساً يحتوي على الموديل
        if isinstance(loaded_obj, dict):
            if 'model' in loaded_obj:
                return loaded_obj['model']
            return loaded_obj # ربما هو قاموس ولكن الموديل هو القاموس نفسه (حالة نادرة)
        return loaded_obj
    except Exception as e:
        print(f"❌ خطأ في تحميل {path}: {e}")
        return None

def extract_features(hand_landmarks):
    """استخراج 60 ميزة (إحداثيات نسبية بدون نقطة المعصم)"""
    points = []
    for lm in hand_landmarks.landmark:
        points.append([lm.x, lm.y, lm.z])

    # النقطة المرجعية (المعصم)
    base_x, base_y, base_z = points[0]
    final_features = []
    
    # نبدأ من 1 (نتجاهل المعصم) ونطرح قيمته من باقي النقط
    for i in range(1, len(points)):
        p = points[i]
        final_features.extend([p[0] - base_x, p[1] - base_y, p[2] - base_z])

    return final_features

def put_arabic_text(img, text, position, color=(0, 255, 0)):
    """رسم نص عربي على الصورة"""
    img_pil = Image.fromarray(img)
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()
    draw.text(position, bidi_text, font=font, fill=color)
    return np.array(img_pil)

# =============================================================
# 🚀 تحميل جميع الموديلات في الذاكرة
# =============================================================
print("🔄 جاري تحميل الموديلات...")
loaded_models = {}
for key, config in MODELS_CONFIG.items():
    print(f"   ... تحميل {config['name']}")
    model = safe_load_model(config['path'])
    if model:
        loaded_models[key] = model
    else:
        print(f"⚠️ فشل تحميل {config['name']}")

if not loaded_models:
    print("❌ لم يتم تحميل أي موديل بنجاح. تأكد من المسارات.")
    exit()

# تعيين الموديل الافتراضي (رقم 4 - Random Forest لأنه الأقوى)
current_key = '4'
current_model = loaded_models.get(current_key)
print(f"✅ تم الجاهزية! الموديل الحالي: {MODELS_CONFIG[current_key]['name']}")
print("💡 اضغط على الأرقام 1, 2, 3, 4 في الكيبورد للتبديل بين الموديلات أثناء التشغيل.")

# =============================================================
# 🎥 إعدادات Mediapipe والكاميرا
# =============================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # قلب الصورة (اختياري)
    # frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # متغيرات العرض
    prediction_text = "..."
    conf_text = ""
    color = (200, 200, 200)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 1. استخراج الميزات
            data = extract_features(hand_landmarks)

            # 2. التوقع باستخدام الموديل الحالي
            if current_model:
                try:
                    prediction = current_model.predict([data])[0]
                    prediction_text = str(prediction)
                    
                    # محاولة حساب الثقة (بعض الموديلات مثل SVC لا تدعمها افتراضياً)
                    if hasattr(current_model, "predict_proba"):
                        probs = current_model.predict_proba([data])[0]
                        confidence = np.max(probs) * 100
                        conf_text = f"({int(confidence)}%)"
                        
                        if confidence < 60:
                            color = (0, 0, 255) # أحمر للثقة المنخفضة
                        else:
                            color = (0, 255, 0) # أخضر
                    else:
                        # في حالة SVC او موديلات لا تدعم الاحتمالات
                        conf_text = "(N/A)" 
                        color = (255, 255, 0) # أصفر

                except Exception as e:
                    prediction_text = "Error"
                    print(f"Predict Error: {e}")

            # رسم النتيجة
            h, w, c = frame.shape
            cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
            
            # خلفية سوداء للنص
            cv2.rectangle(frame, (cx-60, cy-90), (cx+160, cy-30), (0, 0, 0), -1)
            display_str = f"{prediction_text} {conf_text}"
            frame = put_arabic_text(frame, display_str, (cx-50, cy-85), color)

    # =========================================================
    # 🖥️ واجهة العرض (UI)
    # =========================================================
    # رسم شريط الحالة في الأعلى
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    model_name = MODELS_CONFIG[current_key]['name']
    ui_text = f"Current Model: {model_name} (Press 1-4 to switch)"
    cv2.putText(frame, ui_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Sign Language Benchmark', frame)

    # =========================================================
    # ⌨️ التحكم بالكيبورد
    # =========================================================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        new_key = chr(key)
        if new_key in loaded_models:
            current_key = new_key
            current_model = loaded_models[current_key]
            print(f"🔀 Switched to: {MODELS_CONFIG[current_key]['name']}")

cap.release()
cv2.destroyAllWindows()