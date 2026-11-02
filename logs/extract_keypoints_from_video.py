import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Kalau kamu punya Config.py yang sama dengan yang dipakai saat live
# recording, bisa langsung import supaya sequence_length konsisten:
# from Config import sequence_length
# Kalau belum ada / mau override manual, isi langsung di bawah:
sequence_length = 30  # <-- SESUAIKAN dengan yang dipakai di training

# ====== KONFIGURASI ======
RAW_VIDEO_ROOT = r"D:\Projects\SIBI Recognition Lecturers\Datasets"
OUTPUT_KEYPOINTS_ROOT = r"D:\Projects\SIBI Recognition Lecturers\MP_Data"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi")
OVERWRITE_EXISTING = False  # kalau True, sequence yang sudah pernah diekstrak akan diproses ulang
# ==========================

mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    """
    Sama persis dengan fungsi di pipeline live recording kamu.
    pose(132) + left_hand(63) + right_hand(63) = 258, TANPA face.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 4)

    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return results


def get_evenly_spaced_indices(total_frames, n_target):
    """Pilih n_target index frame yang tersebar merata dari 0..total_frames-1."""
    if total_frames <= 0:
        return []
    if total_frames <= n_target:
        # video lebih pendek dari sequence_length -> pakai semua frame yang ada
        # lalu ulang frame terakhir untuk menggenapi jumlahnya
        idx = list(range(total_frames))
        while len(idx) < n_target:
            idx.append(total_frames - 1)
        return idx
    return list(np.linspace(0, total_frames - 1, n_target, dtype=int))


def extract_video_to_keypoints(video_path, save_dir, holistic):
    """Ekstrak satu file video jadi sequence_length file .npy di save_dir."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = set(get_evenly_spaced_indices(total_frames, sequence_length))

    os.makedirs(save_dir, exist_ok=True)

    frame_idx = 0
    saved_idx = 0
    # urutkan indeks target supaya penomoran file .npy tetap 0..sequence_length-1
    sorted_targets = sorted(target_indices)
    target_lookup = {f: i for i, f in enumerate(sorted_targets)}

    while cap.isOpened() and saved_idx < len(sorted_targets):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in target_lookup:
            results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            out_position = target_lookup[frame_idx]
            np.save(os.path.join(save_dir, str(out_position)), keypoints)
            saved_idx += 1

        frame_idx += 1

    cap.release()

    # kalau video lebih pendek dari target (jarang, tapi jaga-jaga), genapi
    # dengan mengulang keypoints frame terakhir yang berhasil diekstrak
    if saved_idx < sequence_length and saved_idx > 0:
        last_kp = np.load(os.path.join(save_dir, f"{saved_idx - 1}.npy"))
        for i in range(saved_idx, sequence_length):
            np.save(os.path.join(save_dir, str(i)), last_kp)


def main():
    if not os.path.isdir(RAW_VIDEO_ROOT):
        print(f"RAW_VIDEO_ROOT tidak ditemukan: {RAW_VIDEO_ROOT}")
        return

    actions = sorted([
        d for d in os.listdir(RAW_VIDEO_ROOT)
        if os.path.isdir(os.path.join(RAW_VIDEO_ROOT, d))
    ])
    print(f"Ditemukan {len(actions)} kelas kata: {actions}\n")

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=False,   # sama seperti live recording: face dimatikan
            enable_segmentation=False) as holistic:

        for action in actions:
            action_dir = os.path.join(RAW_VIDEO_ROOT, action)
            videos = sorted([
                f for f in os.listdir(action_dir)
                if f.lower().endswith(VIDEO_EXTENSIONS)
            ])

            print(f"[{action}] {len(videos)} video ditemukan")

            for seq_num, video_name in enumerate(tqdm(videos, desc=action)):
                save_dir = os.path.join(OUTPUT_KEYPOINTS_ROOT, action, str(seq_num))

                if not OVERWRITE_EXISTING and os.path.isdir(save_dir) and \
                        len(os.listdir(save_dir)) >= sequence_length:
                    continue  # sudah pernah diekstrak, lewati

                video_path = os.path.join(action_dir, video_name)
                extract_video_to_keypoints(video_path, save_dir, holistic)

    print("\nSelesai. Semua keypoints tersimpan di:", OUTPUT_KEYPOINTS_ROOT)


if __name__ == "__main__":
    main()