/**
 * Model Configuration — SIBI Recognition (30 kelas)
 *
 * Model: BiLSTM + Attention (model30class.keras / model30class.h5, dikonversi ke TensorFlow.js)
 * Input  : (None, 30, 258)  -> 30 frame x 258 fitur keypoint
 * Output : (None, 30)       -> 30 label kelas
 */

/* ============================================================================
   SISTEM 15 CLASS (DIKOMENTARI SEMENTARA)
   ============================================================================
export const ACTIONS_15 = [
  'Adik',
  'Apa',
  'Ayah',
  'Baik',
  'Berapa',
  'Bertemu',
  'Cantik',
  'Dari',
  'Dia',
  'Dimana',
  'Ganteng',
  'Gemuk',
  'Hallo',
  'Hobi',
  'Ibu',
] as const;

export const MODEL_PATH_15: string = '/models/tfjs_model15class/model.json';
============================================================================ */

/**
 * SISTEM 30 CLASS (AKTIF DEFAULT)
 * Sesuai public/models/tfjs_model30class/daftar_kelas.md
 */
export const ACTIONS = [
  'Adik',
  'Apa',
  'Ayah',
  'Baik',
  'Berapa',
  'Bertemu',
  'Cantik',
  'Dari',
  'Dia',
  'Dimana',
  'Ganteng',
  'Gemuk',
  'Hallo',
  'Hobi',
  'Ibu',
  'Jumat',
  'Jurusan',
  'Kabar',
  'Kakak',
  'Kalian',
  'Kami',
  'Kamis',
  'Kampus',
  'Kamu',
  'Kelas',
  'Keluarga',
  'Kemana',
  'Kenapa',
  'Kita',
  'Kuliah',
] as const;

/**
 * SISTEM FULL CLASS (63 KELAS)
 * Sesuai public/models/tfjs_modelFullClass/daftar_kelas.md
 */
export const ACTIONS_FULL = [
  'Adik',
  'Apa',
  'Ayah',
  'Baik',
  'Berapa',
  'Bertemu',
  'Cantik',
  'Dari',
  'Dia',
  'Dimana',
  'Ganteng',
  'Gemuk',
  'Hallo',
  'Hobi',
  'Ibu',
  'Jumat',
  'Jurusan',
  'Kabar',
  'Kakak',
  'Kalian',
  'Kami',
  'Kamis',
  'Kampus',
  'Kamu',
  'Kelas',
  'Keluarga',
  'Kemana',
  'Kenapa',
  'Kita',
  'Kuliah',
  'Kurus',
  'Lucu',
  'Malam',
  'Mau',
  'Mereka',
  'Minggu',
  'Nama',
  'Pagi',
  'Pelit',
  'Pendidikan',
  'Pintar',
  'Pulang',
  'Rabu',
  'Ramah',
  'Rendah Hati',
  'Sabar',
  'Sabtu',
  'Sakit',
  'Sampai Jumpa',
  'Saya',
  'Sekolah',
  'Selamat',
  'Selasa',
  'Semester',
  'Senang',
  'Senin',
  'Siang',
  'Siapa',
  'Sore',
  'Suka',
  'Terimakasih',
  'Tinggal',
  'Umur',
] as const;

export const ACTIONS_63 = ACTIONS_FULL;

// Jumlah frame per urutan isyarat (sesuai input model: (None, 30, 258))
export const SEQUENCE_LENGTH: number = 30;

// Batas confidence minimum agar prediksi dianggap valid
export const THRESHOLD: number = 0.8;

// Jumlah prediksi beruntun yang sama sebelum dikunci (stabilisasi)
export const STABILITY_FRAMES: number = 4;

// Path model TensorFlow.js yang ditaruh di public/models/
// export const MODEL_PATH_15: string = '/models/tfjs_model15class/model.json'; // 15 Class
export const MODEL_PATH_30: string = '/models/tfjs_model30class/model.json'; // 30 Class
export const MODEL_PATH_FULL: string = '/models/tfjs_modelFullClass/model.json'; // 63 Class (Full)
export const MODEL_PATH_63: string = MODEL_PATH_FULL;

export const MODEL_PATH: string = MODEL_PATH_30; // Default path

// Konfigurasi keypoint MediaPipe Holistic sesuai ekstraksi fitur saat training
export const KEYPOINTS_CONFIG = {
  POSE_KEYPOINTS: 33,
  POSE_DIMENSIONS: 4, // x, y, z, visibility
  LEFT_HAND_KEYPOINTS: 21,
  RIGHT_HAND_KEYPOINTS: 21,
  HAND_DIMENSIONS: 3, // x, y, z
  TOTAL_FEATURES: 258, // (33*4) + (21*3) + (21*3)
};

// Warna badge probabilitas berdasarkan level confidence
export const PREDICTION_LEVELS = {
  high: { min: 0.7, label: 'Yakin' },
  medium: { min: 0.4, label: 'Ragu' },
  low: { min: 0, label: 'Rendah' },
};
