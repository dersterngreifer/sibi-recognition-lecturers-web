"use client";

import { useMemo, useState, useRef, useEffect, useCallback } from "react";
import {
  Search,
  PlayCircle,
  X,
  Camera,
  CameraOff,
  CheckCircle2,
  Sparkles,
  FlipHorizontal2,
  ChevronLeft,
  ChevronRight,
  Info,
  Layers,
  Cpu,
} from "lucide-react";
import { ACTIONS, SEQUENCE_LENGTH, STABILITY_FRAMES, THRESHOLD } from "@/config/modelConfig";
import {
  initializeMediaPipe,
  detectPoseLandmarks,
  extractKeypoints,
  drawLandmarks,
  isHandDetected,
} from "@/lib/mediapipeUtils";
import { loadModel, predictGesture, isModelLoaded } from "@/lib/modelUtils";

type ActionWord = (typeof ACTIONS)[number];

/* ============================================================================
   KATEGORI KATA - 15 KELAS (DIKOMENTARI SEMENTARA)
   ============================================================================
const WORD_CATEGORIES_15: Record<string, string> = {
  Adik: "Keluarga & Orang",
  Ayah: "Keluarga & Orang",
  Ibu: "Keluarga & Orang",
  Dia: "Keluarga & Orang",
  Hallo: "Sapaan & Tanya",
  Apa: "Sapaan & Tanya",
  Berapa: "Sapaan & Tanya",
  Dimana: "Sapaan & Tanya",
  Bertemu: "Sapaan & Tanya",
  Dari: "Sapaan & Tanya",
  Baik: "Sapaan & Tanya",
  Cantik: "Sifat & Ciri",
  Ganteng: "Sifat & Ciri",
  Gemuk: "Sifat & Ciri",
  Hobi: "Minat",
};
============================================================================ */

/**
 * KATEGORI KATA - 30 KELAS (AKTIF)
 */
const WORD_CATEGORIES: Record<ActionWord, string> = {
  // Keluarga & Orang (10 kata)
  Adik: "Keluarga & Orang",
  Ayah: "Keluarga & Orang",
  Dia: "Keluarga & Orang",
  Ibu: "Keluarga & Orang",
  Kakak: "Keluarga & Orang",
  Kalian: "Keluarga & Orang",
  Kami: "Keluarga & Orang",
  Kamu: "Keluarga & Orang",
  Keluarga: "Keluarga & Orang",
  Kita: "Keluarga & Orang",

  // Sapaan & Tanya (9 kata)
  Apa: "Sapaan & Tanya",
  Berapa: "Sapaan & Tanya",
  Bertemu: "Sapaan & Tanya",
  Dari: "Sapaan & Tanya",
  Dimana: "Sapaan & Tanya",
  Hallo: "Sapaan & Tanya",
  Kabar: "Sapaan & Tanya",
  Kemana: "Sapaan & Tanya",
  Kenapa: "Sapaan & Tanya",

  // Akademik & Kampus (4 kata)
  Jurusan: "Akademik & Kampus",
  Kampus: "Akademik & Kampus",
  Kelas: "Akademik & Kampus",
  Kuliah: "Akademik & Kampus",

  // Waktu & Hari (2 kata)
  Jumat: "Waktu & Hari",
  Kamis: "Waktu & Hari",

  // Sifat & Ciri (4 kata)
  Baik: "Sifat & Ciri",
  Cantik: "Sifat & Ciri",
  Ganteng: "Sifat & Ciri",
  Gemuk: "Sifat & Ciri",

  // Minat (1 kata)
  Hobi: "Minat",
};

const CATEGORIES = [
  "Semua",
  "Keluarga & Orang",
  "Sapaan & Tanya",
  "Akademik & Kampus",
  "Waktu & Hari",
  "Sifat & Ciri",
  "Minat",
] as const;

export default function KosakataPage() {
  const [query, setQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("Semua");

  // State untuk modal latihan interaktif TFJS
  const [activePracticeWord, setActivePracticeWord] = useState<ActionWord | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isMirrored, setIsMirrored] = useState(true);
  const [isModelReady, setIsModelReady] = useState(false);
  const [loadingError, setLoadingError] = useState<string | null>(null);

  // Status deteksi kamera
  const [livePrediction, setLivePrediction] = useState<{
    gesture: string;
    confidence: number;
    isMatch: boolean;
  }>({
    gesture: "Menunggu gerakan...",
    confidence: 0,
    isMatch: false,
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sequenceRef = useRef<Float32Array[]>([]);
  const animFrameIdRef = useRef<number | null>(null);
  const isMirroredRef = useRef(isMirrored);
  const activeWordRef = useRef(activePracticeWord);

  useEffect(() => {
    isMirroredRef.current = isMirrored;
  }, [isMirrored]);

  useEffect(() => {
    activeWordRef.current = activePracticeWord;
  }, [activePracticeWord]);

  // Filter kata berdasarkan pencarian & kategori
  const filteredWords = useMemo(() => {
    const q = query.trim().toLowerCase();
    return ACTIONS.filter((word) => {
      const matchQuery = !q || word.toLowerCase().includes(q);
      const matchCategory =
        selectedCategory === "Semua" || WORD_CATEGORIES[word] === selectedCategory;
      return matchQuery && matchCategory;
    });
  }, [query, selectedCategory]);

  // Mulai kamera saat modal latihan dibuka
  const startPracticeCamera = useCallback(async () => {
    try {
      setLoadingError(null);
      if (!isModelLoaded()) {
        const loaded = await loadModel();
        if (!loaded) throw new Error("Gagal memuat model TensorFlow.js");
      }
      const mpReady = await initializeMediaPipe();
      if (!mpReady) throw new Error("Gagal memuat MediaPipe Holistic");

      setIsModelReady(true);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsCameraActive(true);
      sequenceRef.current = [];
    } catch (err) {
      console.error(err);
      setLoadingError(err instanceof Error ? err.message : "Tidak dapat mengakses kamera");
      setIsCameraActive(false);
    }
  }, []);

  const stopPracticeCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
      animFrameIdRef.current = null;
    }
    setIsCameraActive(false);
    sequenceRef.current = [];
    setLivePrediction({
      gesture: "Menunggu gerakan...",
      confidence: 0,
      isMatch: false,
    });
  }, []);

  // Frame detection loop untuk modal latihan
  const detectPracticeFrame = useCallback(async () => {
    if (
      !videoRef.current ||
      !canvasRef.current ||
      !isCameraActive ||
      videoRef.current.videoWidth === 0
    ) {
      animFrameIdRef.current = requestAnimationFrame(detectPracticeFrame);
      return;
    }

    try {
      const now = performance.now();
      const video = videoRef.current;
      const canvas = canvasRef.current;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      if (!offscreenRef.current) {
        offscreenRef.current = document.createElement("canvas");
      }
      offscreenRef.current.width = video.videoWidth;
      offscreenRef.current.height = video.videoHeight;

      let source: HTMLVideoElement | HTMLCanvasElement = video;
      if (isMirroredRef.current) {
        const ctx = offscreenRef.current.getContext("2d");
        if (ctx) {
          ctx.save();
          ctx.translate(offscreenRef.current.width, 0);
          ctx.scale(-1, 1);
          ctx.drawImage(video, 0, 0);
          ctx.restore();
          source = offscreenRef.current;
        }
      }

      const landmarks = await detectPoseLandmarks(source, now);
      drawLandmarks(canvas, landmarks, true);

      if (landmarks && isHandDetected(landmarks)) {
        const kp = extractKeypoints(landmarks, isMirroredRef.current);
        sequenceRef.current.push(kp);
        sequenceRef.current = sequenceRef.current.slice(-SEQUENCE_LENGTH);

        if (sequenceRef.current.length === SEQUENCE_LENGTH) {
          const res = await predictGesture(sequenceRef.current);
          if (res.predictions.length > 0) {
            const target = activeWordRef.current;
            const isMatch = res.gesture === target && res.confidence >= THRESHOLD;
            setLivePrediction({
              gesture: res.gesture,
              confidence: Math.round(res.confidence * 100),
              isMatch,
            });
          }
        }
      } else {
        sequenceRef.current = [];
        setLivePrediction((prev) => ({
          ...prev,
          gesture: "Tangan belum terdeteksi",
          confidence: 0,
          isMatch: false,
        }));
      }
    } catch (err) {
      console.error("Practice frame error:", err);
    }

    animFrameIdRef.current = requestAnimationFrame(detectPracticeFrame);
  }, [isCameraActive]);

  useEffect(() => {
    if (isCameraActive) {
      animFrameIdRef.current = requestAnimationFrame(detectPracticeFrame);
    }
    return () => {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
    };
  }, [isCameraActive, detectPracticeFrame]);

  // Buka latihan untuk kata tertentu
  const openPractice = (word: ActionWord) => {
    setActivePracticeWord(word);
    startPracticeCamera();
  };

  const closePractice = () => {
    stopPracticeCamera();
    setActivePracticeWord(null);
  };

  const nextWord = () => {
    if (!activePracticeWord) return;
    const curIdx = ACTIONS.indexOf(activePracticeWord);
    const nextIdx = (curIdx + 1) % ACTIONS.length;
    setActivePracticeWord(ACTIONS[nextIdx]);
    sequenceRef.current = [];
  };

  const prevWord = () => {
    if (!activePracticeWord) return;
    const curIdx = ACTIONS.indexOf(activePracticeWord);
    const prevIdx = (curIdx - 1 + ACTIONS.length) % ACTIONS.length;
    setActivePracticeWord(ACTIONS[prevIdx]);
    sequenceRef.current = [];
  };

  return (
    <main className="w-full px-4 sm:px-8 py-6 sm:py-8 max-w-6xl mx-auto">
      {/* Header Halaman */}
      <header className="mb-6 flex flex-col md:flex-row md:items-end md:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-teal/10 text-teal border border-teal/20">
              <Layers className="h-3.5 w-3.5" />
              {/* Model TFJS: 15 Kelas */}
              Model TFJS: 30 Kelas
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
              <Cpu className="h-3.5 w-3.5" />
              BiLSTM + Attention
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-semibold text-ink">Kosakata SIBI</h1>
          <p className="text-ink-soft text-sm mt-1 max-w-2xl">
            {/* Menampilkan seluruh 15 kata isyarat yang telah dilatih pada model BiLSTM. */}
            Menampilkan seluruh 30 kata isyarat yang telah dilatih pada model BiLSTM. Klik tombol{" "}
            <strong>Latihan (TFJS)</strong> pada setiap kata untuk menguji gerakan secara langsung melalui webcam.
          </p>
        </div>

        {/* Input Pencarian */}
        <div className="relative w-full md:w-72">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-ink-soft" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Cari kata isyarat..."
            className="w-full pl-9 pr-9 py-2.5 rounded-xl border border-surface-border bg-surface text-sm text-ink placeholder-ink-soft focus:outline-none focus:border-teal transition-colors shadow-sm"
          />
          {query && (
            <button
              onClick={() => setQuery("")}
              aria-label="Hapus pencarian"
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-ink-soft hover:text-ink"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </header>

      {/* Filter Kategori */}
      <div className="flex flex-wrap items-center gap-2 mb-6">
        {CATEGORIES.map((cat) => {
          const isSelected = selectedCategory === cat;
          return (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                isSelected
                  ? "bg-teal text-white shadow-sm"
                  : "bg-surface border border-surface-border text-ink-soft hover:text-ink hover:border-teal/50"
              }`}
            >
              {cat}
            </button>
          );
        })}
        <span className="text-xs text-ink-soft ml-auto">
          Menampilkan {filteredWords.length} dari {ACTIONS.length} kata
        </span>
      </div>

      {/* Grid Kartu Kosakata */}
      {filteredWords.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-ink-soft gap-2 rounded-2xl border border-dashed border-surface-border bg-surface/50">
          <Search className="h-10 w-10 opacity-30" />
          <p className="font-medium text-sm">Kata &ldquo;{query}&rdquo; tidak ditemukan</p>
          <button
            onClick={() => {
              setQuery("");
              setSelectedCategory("Semua");
            }}
            className="text-xs text-teal hover:underline mt-1"
          >
            Reset filter pencarian
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
          {filteredWords.map((word) => {
            const index = ACTIONS.indexOf(word) + 1;
            const category = WORD_CATEGORIES[word];

            return (
              <div
                key={word}
                className="group relative flex flex-col justify-between rounded-xl border border-surface-border bg-surface p-4 shadow-sm hover:border-teal/40 hover:shadow-md transition-all"
              >
                <div className="flex items-start gap-3">
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-teal-soft text-teal font-display text-lg font-bold">
                    {word.charAt(0)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold text-ink text-base truncate">{word}</h3>
                      <span className="text-[11px] px-2 py-0.5 rounded-md bg-teal/10 text-teal font-medium shrink-0">
                        {category}
                      </span>
                    </div>
                    <p className="text-xs text-ink-soft mt-1 flex items-center gap-1.5">
                      <PlayCircle className="h-3.5 w-3.5 opacity-60" />
                      Urutan ke-{index} dalam model
                    </p>
                  </div>
                  <span className="text-xs font-mono font-medium text-ink-soft shrink-0">
                    #{index.toString().padStart(2, "0")}
                  </span>
                </div>

                <div className="mt-4 pt-3 border-t border-surface-border/60 flex items-center justify-between">
                  <span className="text-[11px] text-ink-soft">BiLSTM Sequence (30 frame)</span>
                  <button
                    onClick={() => openPractice(word)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-teal text-white hover:bg-teal/90 shadow-sm transition-all"
                  >
                    <Sparkles className="h-3.5 w-3.5" />
                    Latihan (TFJS)
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Modal Latihan Interaktif dengan Webcam & TFJS */}
      {activePracticeWord && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs animate-fadeIn">
          <div className="relative w-full max-w-2xl rounded-2xl border border-surface-border bg-surface shadow-2xl overflow-hidden flex flex-col">
            {/* Modal Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-surface-border bg-surface-soft">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-teal text-white">
                  <Sparkles className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="text-base font-semibold text-ink flex items-center gap-2">
                    Latihan Isyarat: <span className="text-teal font-bold">{activePracticeWord}</span>
                  </h3>
                  <p className="text-xs text-ink-soft">
                    {/* {WORD_CATEGORIES[activePracticeWord]} • Model TFJS 15 Kelas */}
                    {WORD_CATEGORIES[activePracticeWord]} • Model TFJS 30 Kelas
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsMirrored((prev) => !prev)}
                  title="Balik Tampilan Kamera"
                  className={`p-2 rounded-lg border text-xs font-medium transition-colors ${
                    isMirrored
                      ? "border-teal/40 bg-teal/10 text-teal"
                      : "border-surface-border text-ink-soft hover:text-ink"
                  }`}
                >
                  <FlipHorizontal2 className="h-4 w-4" />
                </button>
                <button
                  onClick={closePractice}
                  className="p-2 rounded-lg text-ink-soft hover:text-ink hover:bg-surface-border/40 transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </div>

            {/* Modal Body: Kamera & Deteksi */}
            <div className="p-5 flex flex-col gap-4">
              {loadingError ? (
                <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-600 dark:text-rose-400 text-xs">
                  {loadingError}
                </div>
              ) : null}

              {/* Viewport Kamera */}
              <div className="relative aspect-video w-full rounded-xl overflow-hidden bg-black/90 flex items-center justify-center">
                <video
                  ref={videoRef}
                  playsInline
                  muted
                  className={`absolute inset-0 w-full h-full object-cover ${
                    isMirrored ? "-scale-x-100" : ""
                  }`}
                />
                <canvas
                  ref={canvasRef}
                  className="absolute inset-0 w-full h-full object-cover pointer-events-none"
                />

                {!isCameraActive && (
                  <div className="flex flex-col items-center gap-2 text-white/70">
                    <CameraOff className="h-8 w-8" />
                    <span className="text-xs">Menyiapkan kamera dan model TFJS...</span>
                  </div>
                )}

                {/* Badge Status Deteksi Live */}
                {isCameraActive && (
                  <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none">
                    <span className="px-2.5 py-1 rounded-md bg-black/60 backdrop-blur-sm text-[11px] text-white flex items-center gap-1.5">
                      <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                      Live Inference TFJS
                    </span>

                    {livePrediction.isMatch ? (
                      <span className="px-3 py-1 rounded-md bg-emerald-600 text-white text-xs font-semibold flex items-center gap-1.5 shadow-md">
                        <CheckCircle2 className="h-4 w-4" />
                        Cocok! {livePrediction.confidence}%
                      </span>
                    ) : livePrediction.confidence > 0 ? (
                      <span className="px-2.5 py-1 rounded-md bg-black/60 backdrop-blur-sm text-[11px] text-white/90">
                        Terdeteksi: <strong>{livePrediction.gesture}</strong> ({livePrediction.confidence}%)
                      </span>
                    ) : null}
                  </div>
                )}
              </div>

              {/* Feedback Hasil Prediksi */}
              <div
                className={`p-3.5 rounded-xl border flex items-center justify-between transition-colors ${
                  livePrediction.isMatch
                    ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:text-emerald-300"
                    : "bg-surface-soft border-surface-border text-ink"
                }`}
              >
                <div>
                  <div className="text-xs font-medium text-ink-soft">Target Latihan:</div>
                  <div className="text-lg font-bold text-ink">{activePracticeWord}</div>
                </div>

                <div className="text-right">
                  <div className="text-xs font-medium text-ink-soft">Hasil Prediksi Model:</div>
                  <div className="text-sm font-semibold flex items-center gap-1.5 justify-end">
                    {livePrediction.isMatch && <CheckCircle2 className="h-4 w-4 text-emerald-500" />}
                    <span>{livePrediction.gesture}</span>
                    {livePrediction.confidence > 0 && (
                      <span className="text-xs text-ink-soft">({livePrediction.confidence}%)</span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Modal Footer: Navigasi Kata */}
            <div className="px-5 py-3 border-t border-surface-border bg-surface-soft flex items-center justify-between">
              <button
                onClick={prevWord}
                className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-surface-border text-xs font-medium text-ink-soft hover:text-ink hover:bg-surface transition-colors"
              >
                <ChevronLeft className="h-4 w-4" />
                Sebelumnya
              </button>

              <span className="text-xs text-ink-soft font-mono">
                {ACTIONS.indexOf(activePracticeWord) + 1} / {ACTIONS.length}
              </span>

              <button
                onClick={nextWord}
                className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-surface-border text-xs font-medium text-ink-soft hover:text-ink hover:bg-surface transition-colors"
              >
                Selanjutnya
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
