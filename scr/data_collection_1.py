import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
import random

# ---------------- إعدادات محسنة ----------------
DATA_DIR = r"E:\salah_programing\python\open_cv\sign_language\myData"
OUT_CSV = "hands_landmarks_v2_3D_optimized.csv"
MIN_DETECTION_CONF = 0.3
USE_RIGHT_HAND_ONLY = True
FLOAT_FORMAT = "%.12f"
TARGET_PER_LABEL = 100
RANDOM_SEED = 42

# -----------------------------------------------

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# خريطة الحروف (نفس القائمة السابقة)
franco_to_ar = {
    "al": "ال", "alif": "أ", "ayn": "ع", "baa": "ب", "daad": "ض",
    "dal": "د", "dhal": "ذ", "faa": "ف", "ghayn": "غ", "haa": "ح",
    "haa_h": "ه", "hamza_on_yaa": "ئ", "jeem": "ج", "kaf": "ك",
    "khaa": "خ", "laam_alif": "لا", "lam": "ل", "meem": "م",
    "noon": "ن", "qaf": "ق", "raa": "ر", "saad": "ص", "seen": "س",
    "sheen": "ش", "taa": "ت", "taa_h": "ط", "taa_marbuta": "ة",
    "thaa": "ث", "waw": "و", "yaa": "ي", "zaa_h": "ظ", "zay": "ز"
}

ordered_letters = [
    "alif", "baa", "taa", "thaa", "jeem", "haa", "khaa",
    "dal", "dhal", "raa", "zay", "seen", "sheen", "saad",
    "daad", "taa_h", "zaa_h", "ayn", "ghayn", "faa", "qaf",
    "kaf", "lam", "meem", "noon", "haa_h", "waw", "yaa",
    "laam_alif", "al", "hamza_on_yaa", "taa_marbuta"
]

# --- تحسين 1: إضافة العمود Z وتوضيح الأعمدة ---
# الأعمدة ستكون:
# [x0_glob, y0_glob, z0_glob] -> مرجع المعصم (للتصور فقط، يفضل عدم تدريب الموديل عليها)
# [x1_rel, y1_rel, z1_rel ... x20_rel, y20_rel, z20_rel] -> البيانات النسبية الفعلية للتدريب
cols = ["x0_glob", "y0_glob", "z0_glob"]
for i in range(1, 21):
    cols.append(f"x{i}")
    cols.append(f"y{i}")
    cols.append(f"z{i}")
cols.append("label")

# إعداد MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=MIN_DETECTION_CONF)

# المخزن المؤقت
samples = {franco_to_ar.get(k, k): [] for k in ordered_letters}
missing = []

folders = [f for f in ordered_letters if os.path.isdir(os.path.join(DATA_DIR, f))]
print(f"Starting optimized processing with 3D Landmarks (X, Y, Z)...")

for folder_name in folders:
    folder_path = os.path.join(DATA_DIR, folder_name)
    arabic_label = franco_to_ar.get(folder_name, folder_name)
    
    img_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    
    for img_name in tqdm(img_files, desc=f"{arabic_label}", unit="img"):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            missing.append((folder_name, img_name, "read_error"))
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            missing.append((folder_name, img_name, "no_hand"))
            continue

        # منطق اختيار اليد (اليمنى)
        chosen_idx = 0
        if USE_RIGHT_HAND_ONLY and results.multi_handedness:
            found_right = False
            for idx, handedness in enumerate(results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    chosen_idx = idx
                    found_right = True
                    break
            if not found_right:
                # لو لم نجد يداً يمنى وتجاوزنا، نسجلها كمفقودة
                missing.append((folder_name, img_name, "left_hand_ignored"))
                continue

        lm_list = results.multi_hand_landmarks[chosen_idx].landmark

        # --- تحسين 2: استخراج X, Y, Z ---
        # النقطة 0 (المعصم) هي المرجع
        x0, y0, z0 = lm_list[0].x, lm_list[0].y, lm_list[0].z

        # مصفوفة للنقاط الـ 20 المتبقية (بدون المعصم) بـ 3 أبعاد
        # Shape: (20, 3)
        pts_rel = np.zeros((20, 3), dtype=np.float32)

        for i in range(1, 21):
            # حساب الإحداثيات النسبية (Relative Coordinates)
            pts_rel[i-1, 0] = lm_list[i].x - x0
            pts_rel[i-1, 1] = lm_list[i].y - y0
            pts_rel[i-1, 2] = lm_list[i].z - z0  # البعد الثالث النسبي

        # تخزين العينة
        samples[arabic_label].append({
            'global': [x0, y0, z0],
            'pts_rel': pts_rel
        })

hands.close()

# --- تحسين 3: تحديث دالة الـ Augmentation لتدعم 3D ---
def augment_landmarks_3d(pts_rel_20x3):
    """
    pts_rel_20x3: مصفوفة (20, 3) للنقاط النسبية
    """
    pts = pts_rel_20x3.copy()
    
    # 1. Rotation (تدوير حول المحور Z فقط - أي تدوير الصورة 2D)
    # هذا يؤثر على X و Y فقط، بينما Z يبقى نسبياً كما هو تقريباً
    angle_deg = np.random.uniform(-13, 13)
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    
    # تطبيق التدوير على أول عمودين (X, Y)
    pts[:, :2] = pts[:, :2].dot(R.T)
    
    # 2. Scale (تكبير/تصغير اليد كاملة بما فيها العمق)
    scale = np.random.uniform(0.90, 1.10)
    pts = pts * scale
    
    # 3. Jitter (ضوضاء عشوائية) لكل الأبعاد
    noise = np.random.normal(0, 0.002, pts.shape) # ضوضاء خفيفة
    pts = pts + noise
    
    return pts

print("\n--- Balancing Data (Augmentation/Sampling) ---")
out_rows = []

for label, arr_list in samples.items():
    n_exist = len(arr_list)
    
    # إذا لا توجد صور لهذا الحرف
    if n_exist == 0:
        continue

    # استراتيجية الملء (Filling Strategy)
    # 1. إضافة البيانات الحقيقية أولاً
    # 2. إذا زاد عن الحد، نأخذ عينة. إذا نقص، نولد بيانات.
    
    # تحديد العدد المطلوب أخذه من الحقيقي
    take_n = min(n_exist, TARGET_PER_LABEL)
    indices = np.random.choice(range(n_exist), size=take_n, replace=False)
    
    for idx in indices:
        s = arr_list[idx]
        # تجميع الصف: [x0,y0,z0] + [flat 60 points] + [label]
        flat_data = s['global'] + s['pts_rel'].flatten().tolist() + [label]
        out_rows.append(flat_data)
        
    # إذا كنا بحاجة للمزيد (Augmentation)
    needed = TARGET_PER_LABEL - take_n
    if needed > 0:
        added = 0
        tries = 0
        while added < needed and tries < needed * 50:
            tries += 1
            # اختر عينة عشوائية للنسخ منها
            rand_idx = np.random.randint(0, n_exist)
            base_sample = arr_list[rand_idx]
            
            # توليد نقاط جديدة
            aug_pts = augment_landmarks_3d(base_sample['pts_rel'])
            
            # بالنسبة لـ global x0,y0,z0 للعينة المولدة، نأخذ الأصلي مع إزاحة بسيطة جداً
            # (هذا فقط للحفاظ على شكل البيانات، لكن لا يهم الموديل لأننا سنستخدم النسبي)
            aug_global = [
                base_sample['global'][0] + np.random.normal(0, 0.01),
                base_sample['global'][1] + np.random.normal(0, 0.01),
                base_sample['global'][2]  # Z usually doesn't shift much globally in 2D image logic
            ]
            
            flat_data = aug_global + aug_pts.flatten().tolist() + [label]
            out_rows.append(flat_data)
            added += 1
            
    print(f"Label: {label} -> Original: {n_exist} -> Final: {take_n + (needed if needed > 0 else 0)}")

# حفظ الملف النهائي
df_out = pd.DataFrame(out_rows, columns=cols)
df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", float_format=FLOAT_FORMAT)

print("\nSaved optimized CSV:", OUT_CSV)
print("Final Shape:", df_out.shape)

if missing:
    pd.DataFrame(missing, columns=["Folder", "Image", "Reason"]).to_csv("missing_log_v2.csv", index=False)
    print(f"Logged {len(missing)} missing/ignored images.")