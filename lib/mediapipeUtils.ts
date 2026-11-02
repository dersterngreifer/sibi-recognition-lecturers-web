import { KEYPOINTS_CONFIG } from "@/config/modelConfig";
import type { Holistic, Options, Results } from "@mediapipe/holistic";
import type {
  drawConnectors as DrawConnectorsFn,
  drawLandmarks as DrawLandmarksFn,
} from "@mediapipe/drawing_utils";

const MEDIAPIPE_VERSION = "0.5.1675471629";
const DRAWING_UTILS_VERSION = "0.3.1675466124";

const HOLISTIC_SCRIPT_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@${MEDIAPIPE_VERSION}/holistic.js`;
const DRAWING_UTILS_SCRIPT_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@${DRAWING_UTILS_VERSION}/drawing_utils.js`;

type HolisticConstructor = typeof Holistic;
type DrawConnectors = typeof DrawConnectorsFn;
type DrawLandmarks = typeof DrawLandmarksFn;

interface MediaPipeGlobals {
  Holistic?: HolisticConstructor;
  POSE_CONNECTIONS?: [number, number][];
  HAND_CONNECTIONS?: [number, number][];
  drawConnectors?: DrawConnectors;
  drawLandmarks?: DrawLandmarks;
}

let holistic: Holistic | null = null;
let resolveDetection: ((results: Results) => void) | null = null;
let initPromise: Promise<boolean> | null = null;

const loadedScripts = new Set<string>();
const loadingScripts = new Map<string, Promise<void>>();

function getGlobals(): MediaPipeGlobals {
  return (typeof window !== "undefined" ? window : {}) as MediaPipeGlobals;
}

function loadScript(src: string): Promise<void> {
  if (loadedScripts.has(src)) return Promise.resolve();

  const inFlight = loadingScripts.get(src);
  if (inFlight) return inFlight;

  const existing = document.querySelector<HTMLScriptElement>(
    `script[src="${src}"]`,
  );

  const promise = new Promise<void>((resolve, reject) => {
    if (existing) {
      // Script sudah ada di DOM tapi status eksekusinya belum tentu selesai
      // (contoh: panggilan kedua dari React StrictMode saat script pertama
      // masih di-download/dieksekusi). Tunggu event `load` bila masih
      // berjalan; fallback timeout pendek bila sudah dieksekusi.
      let settled = false;
      const done = (err?: unknown) => {
        if (settled) return;
        settled = true;
        loadingScripts.delete(src);
        if (err) reject(err);
        else {
          loadedScripts.add(src);
          resolve();
        }
      };
      const timer = setTimeout(() => done(), 1200);
      existing.addEventListener("load", () => {
        clearTimeout(timer);
        done();
      }, { once: true });
      existing.addEventListener("error", () => {
        clearTimeout(timer);
        done(new Error(`Gagal memuat script MediaPipe: ${src}`));
      }, { once: true });
      return;
    }

    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.onload = () => {
      loadingScripts.delete(src);
      loadedScripts.add(src);
      resolve();
    };
    script.onerror = () => {
      loadingScripts.delete(src);
      reject(new Error(`Gagal memuat script MediaPipe: ${src}`));
    };
    document.body.appendChild(script);
  });

  loadingScripts.set(src, promise);
  return promise;
}

export async function initializeMediaPipe(): Promise<boolean> {
  if (typeof window === "undefined") return false;

  // Instance sudah jadi → tinggal pakai.
  if (holistic) return true;

  // Ada inisialisasi yang masih berjalan → pakai bersama.
  // (Penting: membuat instance Holistic kedua secara paralel akan
  // mengeksekusi ulang WASM Emscripten yang sama dan memicu
  // "Aborted(Module.arguments has been replaced with plain arguments_)"
  // alias error mknod "File exists". React StrictMode/HMR memanggil
  // effect dua kali, jadi mustahil membuat dua instance.)
  if (initPromise) return initPromise;

  initPromise = (async (): Promise<boolean> => {
    try {
      await Promise.all([
        loadScript(DRAWING_UTILS_SCRIPT_URL),
        loadScript(HOLISTIC_SCRIPT_URL),
      ]);

      const HolisticCtor = getGlobals().Holistic;
      if (!HolisticCtor) {
        console.error("Gagal inisialisasi MediaPipe Holistic: Holistic tidak ditemukan");
        return false;
      }

      holistic = new HolisticCtor({
        locateFile: (file: string) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@${MEDIAPIPE_VERSION}/${file}`,
      });

      const options: Options = {
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
        refineFaceLandmarks: false,
        enableSegmentation: false,
        modelComplexity: 1,
      };
      holistic.setOptions(options);

      holistic.onResults((results) => {
        if (resolveDetection) {
          resolveDetection(results);
          resolveDetection = null;
        }
      });

      // Timeout agar UI tidak menggantung bila WASM gagal/buntu saat load.
      await Promise.race([
        holistic.initialize(),
        new Promise<never>((_, reject) =>
          setTimeout(
            () => reject(new Error("Timeout inisialisasi MediaPipe (WASM)")),
            15000,
          ),
        ),
      ]);

      return true;
    } catch (error) {
      console.error("Gagal inisialisasi MediaPipe Holistic:", error);
      holistic = null;
      return false;
    } finally {
      initPromise = null;
    }
  })();

  return initPromise;
}

export async function detectPoseLandmarks(
  video: HTMLVideoElement | HTMLCanvasElement,
  _timestamp: number,
): Promise<Results | null> {
  if (!holistic) return null;

  try {
    return new Promise((resolve) => {
      resolveDetection = resolve;

      holistic!.send({ image: video }).catch(() => {
        if (resolveDetection === resolve) {
          resolveDetection = null;
          resolve(null as unknown as Results);
        }
      });

      setTimeout(() => {
        if (resolveDetection === resolve) {
          resolveDetection = null;
          resolve(null as unknown as Results);
        }
      }, 1000);
    });
  } catch (error) {
    console.error("Gagal mendeteksi landmark:", error);
    return null;
  }
}

/**
 * Mengubah hasil deteksi MediaPipe menjadi vektor fitur.
 * Panjang & susunan vektor bergantung pada KEYPOINTS_CONFIG di modelConfig.ts.
 *
 * @param mirrored  Saat kamera membalik (mirror), x harus di-unflip dan
 *                  tangan kiri/kanan harus ditukar agar sesuai dengan
 *                  koordinat training (rekaman tanpa mirror).
 */
export function extractKeypoints(
  detectionResult: Results,
  mirrored: boolean = false,
): Float32Array {
  const total = KEYPOINTS_CONFIG.TOTAL_FEATURES;
  if (!total) {
    console.warn("KEYPOINTS_CONFIG.TOTAL_FEATURES belum dikonfigurasi.");
    return new Float32Array(0);
  }

  const keypoints = new Float32Array(total);
  let i = 0;

  try {
    const flipX = (v: number) => (mirrored ? 1 - v : v);

    if (detectionResult.poseLandmarks && KEYPOINTS_CONFIG.POSE_KEYPOINTS) {
      for (let p = 0; p < KEYPOINTS_CONFIG.POSE_KEYPOINTS; p++) {
        const lm = detectionResult.poseLandmarks[p];
        keypoints[i++] = lm ? flipX(lm.x) : 0;
        keypoints[i++] = lm?.y ?? 0;
        keypoints[i++] = lm?.z ?? 0;
        keypoints[i++] = lm?.visibility ?? 0;
      }
    } else {
      i += (KEYPOINTS_CONFIG.POSE_KEYPOINTS || 33) * KEYPOINTS_CONFIG.POSE_DIMENSIONS;
    }

    // Saat mirror: MediaPipe "left" = tangan KANAN user, dan sebaliknya.
    // Un-mirror dengan menukar blok tangan + flip x.
    const leftHandLandmarks = mirrored
      ? detectionResult.rightHandLandmarks
      : detectionResult.leftHandLandmarks;
    const rightHandLandmarks = mirrored
      ? detectionResult.leftHandLandmarks
      : detectionResult.rightHandLandmarks;

    if (leftHandLandmarks && KEYPOINTS_CONFIG.LEFT_HAND_KEYPOINTS) {
      for (let p = 0; p < KEYPOINTS_CONFIG.LEFT_HAND_KEYPOINTS; p++) {
        const lm = leftHandLandmarks[p];
        keypoints[i++] = lm ? flipX(lm.x) : 0;
        keypoints[i++] = lm?.y ?? 0;
        keypoints[i++] = lm?.z ?? 0;
      }
    } else {
      i += (KEYPOINTS_CONFIG.LEFT_HAND_KEYPOINTS || 21) * KEYPOINTS_CONFIG.HAND_DIMENSIONS;
    }

    if (rightHandLandmarks && KEYPOINTS_CONFIG.RIGHT_HAND_KEYPOINTS) {
      for (let p = 0; p < KEYPOINTS_CONFIG.RIGHT_HAND_KEYPOINTS; p++) {
        const lm = rightHandLandmarks[p];
        keypoints[i++] = lm ? flipX(lm.x) : 0;
        keypoints[i++] = lm?.y ?? 0;
        keypoints[i++] = lm?.z ?? 0;
      }
    } else {
      i += (KEYPOINTS_CONFIG.RIGHT_HAND_KEYPOINTS || 21) * KEYPOINTS_CONFIG.HAND_DIMENSIONS;
    }

    // --- Visibility fallback ---
    // JS MediaPipe legacy kadang tidak mengisi visibility untuk pose landmark.
    // Tanpa visibility, model hanya ~13% akurat (dibuktikan via Node benchmark).
    // Fallback: isi rata-rata visibility per-landmark dari data training asli.
    fillVisibilityFallbackIfEmpty(keypoints);

    return keypoints;
  } catch (error) {
    console.error("Gagal mengekstrak keypoints:", error);
    return keypoints;
  }
}

/**
 * Jika SEMUA pose visibility = 0 / undefined (JS legacy tidak menyediakan),
 * isi dengan rata-rata per-landmark dari data training.
 * Distribusi ini terbentuk dari 1270 sequence di Dataset Keypoints/25 kelas.
 *
 * Landmark 0-12  (wajah/bahu/torso): ~1.0 (selalu terlihat)
 * Landmark 13-24 (lengan/siku):       0.86 – 0.98
 * Landmark 25-26 (pinggul):           0.29 – 0.32
 * Landmark 27-32 (kaki):              0.01 – 0.02 (sering di luar frame)
 */
const TRAINING_VIS_MEAN = [
  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
  0.98, 0.99, 0.96, 0.96, 0.95, 0.93, 0.95, 0.94, 0.95, 0.93, 0.86,
  0.86, 0.32, 0.29, 0.02, 0.02, 0.02, 0.02, 0.01, 0.02,
];

// TEMP DEBUG - hapus setelah verifikasi selesai
// Diagnostik nilai visibility: apakah bervariasi per frame/per landmark, atau konstan.
const TEMP_POSE_N = 33;
const TEMP_DEBUG_INTERVAL = 30;
const debugVis = {
  frames: 0,
  real: 0,
  fallback: 0,
  globalMin: 1,
  globalMax: 0,
  distinct: new Set<number>(),
  landmarkSum: new Array<number>(TEMP_POSE_N).fill(0),
  lastRawVis: [] as number[],
};

function fillVisibilityFallbackIfEmpty(keypoints: Float32Array): void {
  const POSE_DIM = KEYPOINTS_CONFIG.POSE_DIMENSIONS; // 4
  let hasRealVis = false;

  let frameMin = 1;
  let frameMax = 0;
  let frameSum = 0;
  let frameAbove = 0;

  for (let p = 0; p < KEYPOINTS_CONFIG.POSE_KEYPOINTS; p++) {
    const v = keypoints[p * POSE_DIM + 3];
    if (v > 0.15) hasRealVis = true;
    frameMin = Math.min(frameMin, v);
    frameMax = Math.max(frameMax, v);
    frameSum += v;
    if (v > 0.15) frameAbove++;
    debugVis.landmarkSum[p] += v;
    debugVis.distinct.add(Math.round(v * 100) / 100);
  }
  debugVis.lastRawVis = Array.from(
    { length: TEMP_POSE_N },
    (_, p) => keypoints[p * POSE_DIM + 3],
  );

  debugVis.frames++;
  if (hasRealVis) debugVis.real++;
  else debugVis.fallback++;
  debugVis.globalMin = Math.min(debugVis.globalMin, frameMin);
  debugVis.globalMax = Math.max(debugVis.globalMax, frameMax);

  // TEMP DEBUG - throttle log tiap 30 frame agar tidak flood console
  if (debugVis.frames % TEMP_DEBUG_INTERVAL === 0) {
    const pctFallback = ((debugVis.fallback / debugVis.frames) * 100).toFixed(1);
    const landmarkMeans = debugVis.landmarkSum
      .slice(0, TEMP_POSE_N)
      .map((s) => (s / debugVis.frames).toFixed(2))
      .join(",");
    console.log(
      `[TEMP DEBUG visibility] frames=${debugVis.frames} fallback=${pctFallback}% | ` +
        `frame: min=${frameMin.toFixed(2)} max=${frameMax.toFixed(2)} mean=${(frameSum / TEMP_POSE_N).toFixed(3)} >0.15=${frameAbove}/${TEMP_POSE_N} | ` +
        `global: min=${debugVis.globalMin.toFixed(2)} max=${debugVis.globalMax.toFixed(2)} distinctVals=${debugVis.distinct.size} | ` +
        `landmark-vis-means=[${landmarkMeans}]`,
    );
  }

  if (hasRealVis) return;

  for (let p = 0; p < KEYPOINTS_CONFIG.POSE_KEYPOINTS; p++) {
    keypoints[p * POSE_DIM + 3] = TRAINING_VIS_MEAN[p];
  }
}

export function drawLandmarks(
  canvas: HTMLCanvasElement,
  detectionResult: Results | null,
  showLandmarks: boolean,
) {
  const ctx = canvas?.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!showLandmarks || !detectionResult) return;

  const globals = getGlobals();
  const { POSE_CONNECTIONS, HAND_CONNECTIONS, drawConnectors } = globals;
  if (!drawConnectors) return;

  ctx.save();

  if (detectionResult.poseLandmarks && POSE_CONNECTIONS) {
    drawConnectors(ctx, detectionResult.poseLandmarks, POSE_CONNECTIONS, {
      color: "rgba(31,111,107,0.55)",
      lineWidth: 1,
    });
    globals.drawLandmarks?.(ctx, detectionResult.poseLandmarks, {
      color: "#1F6F6B",
      radius: 1,
    });
  }

  if (detectionResult.leftHandLandmarks && HAND_CONNECTIONS) {
    drawConnectors(ctx, detectionResult.leftHandLandmarks, HAND_CONNECTIONS, {
      color: "rgba(232,163,61,0.7)",
      lineWidth: 2,
    });
    globals.drawLandmarks?.(ctx, detectionResult.leftHandLandmarks, {
      color: "#E8A33D",
      radius: 3,
    });
  }

  if (detectionResult.rightHandLandmarks && HAND_CONNECTIONS) {
    drawConnectors(ctx, detectionResult.rightHandLandmarks, HAND_CONNECTIONS, {
      color: "rgba(232,163,61,0.7)",
      lineWidth: 2,
    });
    globals.drawLandmarks?.(ctx, detectionResult.rightHandLandmarks, {
      color: "#E8A33D",
      radius: 3,
    });
  }

  ctx.restore();
}

export function isHandDetected(detectionResult: Results | null): boolean {
  if (!detectionResult) return false;
  return !!detectionResult.leftHandLandmarks || !!detectionResult.rightHandLandmarks;
}

export function isPoseDetected(detectionResult: Results | null): boolean {
  if (!detectionResult) return false;
  return !!detectionResult.poseLandmarks;
}

/** Estimasi jarak badan ke kamera dari lebar bahu (asumsi focal length ~700px). */
export function calculateShoulderDistance(
  detectionResult: Results | null,
  imageWidth: number,
  imageHeight: number,
  focalLength: number = 700,
): number {
  if (!detectionResult?.poseLandmarks) return 0;

  const left = detectionResult.poseLandmarks[11];
  const right = detectionResult.poseLandmarks[12];
  if (!left || !right) return 0;

  const x1 = left.x * imageWidth;
  const y1 = left.y * imageHeight;
  const x2 = right.x * imageWidth;
  const y2 = right.y * imageHeight;

  const wPixel = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
  if (wPixel < 10) return 0;

  const SHOULDER_WIDTH_CM = 40;
  return Math.round((SHOULDER_WIDTH_CM * focalLength) / wPixel);
}