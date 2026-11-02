"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  initializeMediaPipe,
  detectPoseLandmarks,
  extractKeypoints,
  drawLandmarks,
  isHandDetected,
  calculateShoulderDistance,
} from "@/lib/mediapipeUtils";
import { loadModel, predictGesture, isModelLoaded, disposeModel } from "@/lib/modelUtils";
import {
  ACTIONS_FULL as ACTIONS,
  MODEL_PATH_FULL as MODEL_PATH,
  SEQUENCE_LENGTH,
  THRESHOLD,
  KEYPOINTS_CONFIG,
} from "@/config/modelConfig";
import {
  Camera, CameraOff, FlipHorizontal2, Gamepad2,
  CheckCircle2, Trophy, RotateCcw, Shuffle, ListChecks, History, Target, Circle, Radar, SkipForward, Sparkles
} from "lucide-react";

const MODEL_READY = Boolean(SEQUENCE_LENGTH && THRESHOLD !== null);

// ====== KONFIGURASI CAPTURE GESTURE ======
// Konsep: training memakai satu video utuh per gesture, lalu 30 frame
// diambil merata dari awal-akhir video (lihat extract_keypoints_from_video.py,
// fungsi get_evenly_spaced_indices). Supaya live inference konsisten dengan
// itu, kita TIDAK lagi memprediksi tiap frame dari sliding window mentah.
// Sebaliknya: rekam seluruh gerakan dari saat tangan muncul sampai selesai,
// baru padatkan (downsample merata) jadi 30 frame, baru prediksi SEKALI.
const HAND_LOST_GRACE_FRAMES = 6; // frame berturut-turut tanpa tangan → gesture dianggap selesai
const MIN_CAPTURE_FRAMES = 10;    // capture lebih pendek dari ini dianggap noise, diabaikan
const MAX_CAPTURE_FRAMES = 90;    // batas atas jaga-jaga kalau tangan terus terdeteksi tanpa jeda

// Prediksi LIVE untuk kartu tampilan saat gesture masih berjalan (hanya umpan
// balik visual — skor tetap dari finalizeCapture).
const LIVE_PREDICT_INTERVAL_MS = 140; // seberapa sering menjalankan prediksi live
const LIVE_STABILITY = 3;             // kelas yang sama N kali berturut-turut baru ditampilkan (anti-kedip)

type Question = {
  word: string;
  originalIndex: number;
  isAnswered: boolean;
  isSkipped?: boolean;
  accuracy: number;
  score: number;
};

type GameState = {
  questions: Question[];
  currentIndex: number;
  isFinished: boolean;
};

type SessionHistory = {
  id: string;
  date: string;
  score: number;
  accuracy: number;
  answeredCount: number;
  skippedCount?: number;
  totalQuestions: number;
};

/**
 * Meniru np.linspace(0, totalFrames - 1, nTarget, dtype=int) dari Python.
 * dtype=int di numpy MEMOTONG desimal (truncate), bukan membulatkan —
 * makanya di sini pakai Math.trunc, bukan Math.round, supaya hasil index
 * yang dipilih identik dengan yang dipakai saat ekstraksi dataset training.
 */
function getEvenlySpacedIndices(totalFrames: number, nTarget: number): number[] {
  if (totalFrames <= 0) return [];
  if (totalFrames <= nTarget) {
    const idx = Array.from({ length: totalFrames }, (_, i) => i);
    while (idx.length < nTarget) idx.push(totalFrames - 1);
    return idx;
  }
  if (nTarget === 1) return [0];
  const step = (totalFrames - 1) / (nTarget - 1);
  return Array.from({ length: nTarget }, (_, i) => Math.trunc(i * step));
}

function downsampleEvenly(frames: Float32Array[], nTarget: number): Float32Array[] {
  const indices = getEvenlySpacedIndices(frames.length, nTarget);
  return indices.map((idx) => frames[idx]);
}

export default function TebakKataPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isMirrored, setIsMirrored] = useState(true);
  const [distance, setDistance] = useState(0);

  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [initError, setInitError] = useState<string | null>(null);
  const [stats, setStats] = useState({ fps: 0 });

  const [prediction, setPrediction] = useState({ gesture: "Menunggu...", confidence: 0, detected: false, isLive: false });
  const [isCapturing, setIsCapturing] = useState(false);

  // --- Game State Management ---
  const [gameState, setGameState] = useState<GameState>({ questions: [], currentIndex: 0, isFinished: false });
  const [history, setHistory] = useState<SessionHistory[]>([]);
  const [showSuccessOverlay, setShowSuccessOverlay] = useState(false);

  // Initialize from LocalStorage or create new
  useEffect(() => {
    const savedHistory = localStorage.getItem("tebakKataHistory");
    if (savedHistory) setHistory(JSON.parse(savedHistory));

    const savedGame = localStorage.getItem("tebakKataState");
    if (savedGame) {
      try {
        const parsed = JSON.parse(savedGame);
        if (parsed && Array.isArray(parsed.questions) && parsed.questions.length === ACTIONS.length) {
          setGameState(parsed);
        } else {
          startNewGame(false, false);
        }
      } catch {
        startNewGame(false, false);
      }
    } else {
      startNewGame(false, false);
    }
  }, []);

  // Save game state whenever it changes
  useEffect(() => {
    if (gameState.questions.length > 0) {
      localStorage.setItem("tebakKataState", JSON.stringify(gameState));
    }
  }, [gameState]);

  const saveCurrentSessionToHistory = (currentState: GameState) => {
    const answeredCount = currentState.questions.filter(q => q.isAnswered).length;
    const skippedCount = currentState.questions.filter(q => q.isSkipped).length;
    if (answeredCount === 0 && skippedCount === 0) return; // don't save empty sessions

    const totalScore = currentState.questions.reduce((sum, q) => sum + q.score, 0);
    const avgAccuracy = answeredCount > 0
      ? Math.round(currentState.questions.reduce((sum, q) => sum + q.accuracy, 0) / answeredCount)
      : 0;

    const newSession: SessionHistory = {
      id: `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
      date: new Date().toLocaleString("id-ID", { dateStyle: "short", timeStyle: "short" }),
      score: totalScore,
      accuracy: avgAccuracy,
      answeredCount,
      skippedCount,
      totalQuestions: currentState.questions.length,
    };

    setHistory(prev => {
      const updated = [newSession, ...prev];
      localStorage.setItem("tebakKataHistory", JSON.stringify(updated));
      return updated;
    });
  };

  const startNewGame = (shuffle: boolean, savePrevious: boolean = true) => {
    if (savePrevious) {
      saveCurrentSessionToHistory(gameState);
    }

    let initialQuestions: Question[] = ACTIONS.map((word, index) => ({
      word,
      originalIndex: index,
      isAnswered: false,
      isSkipped: false,
      accuracy: 0,
      score: 0,
    }));

    if (shuffle) {
      initialQuestions = initialQuestions.sort(() => Math.random() - 0.5);
    }

    setGameState({
      questions: initialQuestions,
      currentIndex: 0,
      isFinished: false,
    });
    setShowSuccessOverlay(false);
  };

  const handleReset = () => {
    if (confirm("Apakah Anda yakin ingin mengulang permainan? Progres saat ini akan disimpan ke riwayat.")) {
      startNewGame(false);
    }
  };

  const handleShuffleReset = () => {
    if (confirm("Apakah Anda yakin ingin mengacak ulang soal? Progres saat ini akan disimpan ke riwayat.")) {
      startNewGame(true);
    }
  };

  const activeQuestion = gameState.questions[gameState.currentIndex];

  // Calculate running totals
  const totalScore = gameState.questions.reduce((sum, q) => sum + q.score, 0);
  const answeredQuestions = gameState.questions.filter(q => q.isAnswered);
  const skippedQuestions = gameState.questions.filter(q => q.isSkipped);
  const avgAccuracy = answeredQuestions.length > 0
    ? Math.round(answeredQuestions.reduce((sum, q) => sum + q.accuracy, 0) / answeredQuestions.length)
    : 0;

  // Apakah tebakan model cocok dengan kata target saat ini (untuk kartu Prediksi Model)
  const predictionMatches =
    !gameState.isFinished &&
    !!activeQuestion &&
    prediction.detected &&
    prediction.gesture.toLowerCase() === activeQuestion.word.toLowerCase();

  // --- Camera & Detection Logic ---
  const isMirroredRef = useRef(isMirrored);
  const rawBufferRef = useRef<Float32Array[]>([]);
  const isCapturingRef = useRef(false);
  const isFinalizingRef = useRef(false);
  const noHandStreakRef = useRef(0);
  const lastFrameTimeRef = useRef(0);
  const animIdRef = useRef<number | null>(null);
  const lastUiUpdateRef = useRef(0);
  const successTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const liveClassBufferRef = useRef<number[]>([]);
  const lastLivePredictRef = useRef(0);
  const liveInFlightRef = useRef(false);

  useEffect(() => { isMirroredRef.current = isMirrored; }, [isMirrored]);

  useEffect(() => {
    if (!offscreenRef.current && typeof document !== "undefined") {
      offscreenRef.current = document.createElement("canvas");
    }

    const init = async () => {
      try {
        setIsLoadingModels(true);
        setInitError(null);
        const ok = await initializeMediaPipe();
        if (!ok) throw new Error("Gagal menginisialisasi kamera/MediaPipe");
        const modelOk = await loadModel(MODEL_PATH, ACTIONS);
        if (!modelOk) throw new Error("Gagal memuat model prediksi (TFJS). Lihat console untuk detail.");
        setIsLoadingModels(false);
      } catch (err) {
        setInitError(err instanceof Error ? err.message : "Terjadi kesalahan");
        setIsLoadingModels(false);
      }
    };
    init();

    return () => {
      disposeModel();
      if (animIdRef.current) cancelAnimationFrame(animIdRef.current);
      if (successTimeoutRef.current) clearTimeout(successTimeoutRef.current);
    };
  }, []);

  const startCamera = async () => {
    try {
      if (!isModelLoaded(MODEL_PATH)) loadModel(MODEL_PATH, ACTIONS).catch(() => {});
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      });
      setStream(mediaStream);
      if (videoRef.current) videoRef.current.srcObject = mediaStream;
      setIsCameraOn(true);
      rawBufferRef.current = [];
      isCapturingRef.current = false;
      noHandStreakRef.current = 0;
      setIsCapturing(false);
    } catch (err) {
      console.error(err);
      alert("Tidak dapat mengakses kamera. Pastikan izin kamera sudah diberikan.");
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      setStream(null);
      if (videoRef.current) videoRef.current.srcObject = null;
      setIsCameraOn(false);
      setDistance(0);
      if (animIdRef.current) cancelAnimationFrame(animIdRef.current);
      rawBufferRef.current = [];
      isCapturingRef.current = false;
      noHandStreakRef.current = 0;
      setIsCapturing(false);
    }
  };

  useEffect(() => () => stream?.getTracks().forEach((t) => t.stop()), [stream]);

  const handleCorrectGuess = useCallback((confidence: number) => {
    if (showSuccessOverlay || gameState.isFinished) return;

    const accuracy = Math.round(confidence * 100);
    // Score is 1000 base * confidence multiplier
    const scoreEarned = Math.round(1000 * confidence);

    setShowSuccessOverlay(true);

    setGameState(prev => {
        const newQuestions = [...prev.questions];
        newQuestions[prev.currentIndex] = {
            ...newQuestions[prev.currentIndex],
            isAnswered: true,
            accuracy,
            score: scoreEarned
        };

        const nextIndex = prev.currentIndex + 1;
        const isFinished = nextIndex >= newQuestions.length;

        // Auto-save to history immediately if finished
        if (isFinished) {
            saveCurrentSessionToHistory({ questions: newQuestions, currentIndex: nextIndex, isFinished });
        }

        return {
            questions: newQuestions,
            currentIndex: isFinished ? prev.currentIndex : nextIndex,
            isFinished
        };
    });

    if (successTimeoutRef.current) clearTimeout(successTimeoutRef.current);
    successTimeoutRef.current = setTimeout(() => {
      setShowSuccessOverlay(false);
    }, 2000);
  }, [showSuccessOverlay, gameState.isFinished]);

  const handleSkip = useCallback(() => {
    if (showSuccessOverlay || gameState.isFinished || !activeQuestion) return;

    // Reset buffer capture agar gerakan kata sebelumnya tidak terbawa ke kata berikutnya
    rawBufferRef.current = [];
    isCapturingRef.current = false;
    noHandStreakRef.current = 0;
    liveClassBufferRef.current = [];
    setIsCapturing(false);
    setPrediction({ gesture: "Menunggu...", confidence: 0, detected: false, isLive: false });

    setGameState((prev) => {
      if (prev.isFinished || prev.currentIndex >= prev.questions.length) return prev;

      const newQuestions = [...prev.questions];
      newQuestions[prev.currentIndex] = {
        ...newQuestions[prev.currentIndex],
        isAnswered: false,
        isSkipped: true,
        accuracy: 0,
        score: 0,
      };

      const nextIndex = prev.currentIndex + 1;
      const isFinished = nextIndex >= newQuestions.length;

      // Auto-save ke riwayat jika semua soal sudah selesai
      if (isFinished) {
        saveCurrentSessionToHistory({ questions: newQuestions, currentIndex: nextIndex, isFinished });
      }

      return {
        questions: newQuestions,
        currentIndex: isFinished ? prev.currentIndex : nextIndex,
        isFinished,
      };
    });
  }, [showSuccessOverlay, gameState.isFinished, activeQuestion]);

  /**
   * Dipanggil saat satu gesture selesai direkam (tangan hilang cukup lama,
   * atau capture sudah mentok MAX_CAPTURE_FRAMES). Memadatkan frame mentah
   * jadi SEQUENCE_LENGTH (30) frame merata — sama seperti proses
   * get_evenly_spaced_indices di extract_keypoints_from_video.py — lalu
   * memanggil model SEKALI untuk seluruh gesture tersebut.
   */
  const finalizeCapture = useCallback(async () => {
    const frames = rawBufferRef.current;
    rawBufferRef.current = [];
    isCapturingRef.current = false;
    noHandStreakRef.current = 0;
    liveClassBufferRef.current = [];
    setIsCapturing(false);

    if (frames.length < MIN_CAPTURE_FRAMES) {
      // Terlalu pendek, kemungkinan besar noise (tangan lewat sekilas)
      setPrediction((prev) => ({ ...prev, detected: false, gesture: "Menunggu tangan..." }));
      return;
    }

    if (isFinalizingRef.current) return;
    isFinalizingRef.current = true;

    try {
      const sequence = downsampleEvenly(frames, SEQUENCE_LENGTH as number);
      const res = await predictGesture(sequence, ACTIONS);

      if (res.predictions.length === 0) return;

      const idx = res.predictions[0];
      const guessedWord = ACTIONS[idx];

      setPrediction({
        gesture: guessedWord,
        confidence: res.confidence,
        detected: true,
        isLive: false,
      });

      if (
        !showSuccessOverlay &&
        !gameState.isFinished &&
        activeQuestion &&
        res.confidence > (THRESHOLD as number) &&
        guessedWord.toLowerCase() === activeQuestion.word.toLowerCase()
      ) {
        handleCorrectGuess(res.confidence);
      }
    } catch (err) {
      console.error("Gagal finalisasi capture:", err);
    } finally {
      isFinalizingRef.current = false;
    }
  }, [showSuccessOverlay, gameState.isFinished, activeQuestion, handleCorrectGuess]);

  /**
   * Prediksi LIVE untuk kartu tampilan saat gesture masih berlangsung.
   * Memakai sliding-window 30 frame terakhir (bukan downsample penuh) — hasilnya
   * hanya umpan balik visual; skor & tebakan akhir tetap dari finalizeCapture.
   * Diseimbangkan dengan buffer kelas beruntun supaya tampilan tidak berkedip.
   */
  const runLivePredict = useCallback(async (frames: Float32Array[]) => {
    try {
      const window = frames.slice(-(SEQUENCE_LENGTH as number));
      if (window.length < (SEQUENCE_LENGTH as number)) return;
      const res = await predictGesture(window, ACTIONS);
      if (res.predictions.length === 0) return;

      const idx = res.predictions[0];
      liveClassBufferRef.current.push(idx);
      liveClassBufferRef.current = liveClassBufferRef.current.slice(-LIVE_STABILITY);

      if (
        !isFinalizingRef.current &&
        liveClassBufferRef.current.length === LIVE_STABILITY &&
        liveClassBufferRef.current.every((p) => p === idx)
      ) {
        const guessedWord = ACTIONS[idx];
        setPrediction((prev) =>
          prev.detected && prev.gesture === guessedWord
            ? { ...prev, confidence: res.confidence }
            : { gesture: guessedWord, confidence: res.confidence, detected: true, isLive: true },
        );
      }
    } catch {
      // prediksi live tidak wajib berhasil — biarkan frame berikutnya mencoba lagi
    }
  }, []);

  const detectFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isCameraOn || videoRef.current.videoWidth === 0) {
      animIdRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    try {
      const now = performance.now();
      const delta = now - lastFrameTimeRef.current;
      lastFrameTimeRef.current = now;

      if (delta > 0 && now - lastUiUpdateRef.current > 300) {
        setStats({ fps: Math.round(1000 / delta) });
        lastUiUpdateRef.current = now;
      }

      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;

      let source: HTMLVideoElement | HTMLCanvasElement = videoRef.current;

      if (isMirroredRef.current && offscreenRef.current) {
        if (offscreenRef.current.width !== videoRef.current.videoWidth) {
          offscreenRef.current.width = videoRef.current.videoWidth;
          offscreenRef.current.height = videoRef.current.videoHeight;
        }
        const ctx = offscreenRef.current.getContext("2d");
        if (ctx) {
          ctx.save();
          ctx.translate(offscreenRef.current.width, 0);
          ctx.scale(-1, 1);
          ctx.drawImage(videoRef.current, 0, 0, offscreenRef.current.width, offscreenRef.current.height);
          ctx.restore();
          source = offscreenRef.current;
        }
      }

      const result = await detectPoseLandmarks(source, now);

      if (result) {
        const dist = calculateShoulderDistance(result, canvasRef.current.width, canvasRef.current.height);
        setDistance((prev) => (Math.abs(prev - dist) > 2 ? dist : prev));
        drawLandmarks(canvasRef.current, result, true);
      } else {
        setDistance(0);
        drawLandmarks(canvasRef.current, null, true);
      }

      const handsDetected = result ? isHandDetected(result) : false;
      const canCapture = MODEL_READY && !showSuccessOverlay && !gameState.isFinished && !!activeQuestion;

      if (canCapture && result) {
        if (handsDetected) {
          noHandStreakRef.current = 0;

          if (!isCapturingRef.current) {
            // Tangan baru muncul → mulai rekam gesture baru.
            // Jangan menimpa tebakan yang sedang tampil — biarkan prediksi live
            // yang memperbaruinya supaya kartu tidak berkedip.
            isCapturingRef.current = true;
            rawBufferRef.current = [];
            liveClassBufferRef.current = [];
            setIsCapturing(true);
            setPrediction((prev) =>
              prev.detected ? prev : { ...prev, detected: true, gesture: "Merekam gerakan..." },
            );
          }

          rawBufferRef.current.push(extractKeypoints(result, isMirroredRef.current));

          // Prediksi live untuk tampilan kartu — ditampilkan saat gesture berjalan.
          if (
            !isFinalizingRef.current &&
            !liveInFlightRef.current &&
            rawBufferRef.current.length >= (SEQUENCE_LENGTH as number) &&
            performance.now() - lastLivePredictRef.current > LIVE_PREDICT_INTERVAL_MS
          ) {
            lastLivePredictRef.current = performance.now();
            liveInFlightRef.current = true;
            const snap = rawBufferRef.current;
            void runLivePredict(snap).finally(() => {
              liveInFlightRef.current = false;
            });
          }

          if (rawBufferRef.current.length >= MAX_CAPTURE_FRAMES) {
            // Sudah kepanjangan, paksa selesaikan supaya tidak menunggu selamanya
            void finalizeCapture();
          }
        } else if (isCapturingRef.current) {
          // Tangan sempat hilang di tengah gesture — tetap simpan frame ini
          // (mengikuti perilaku extract_keypoints Python yang tetap memproses
          // tiap frame video walau salah satu tangan sedang tidak terlihat).
          rawBufferRef.current.push(extractKeypoints(result, isMirroredRef.current));
          noHandStreakRef.current += 1;

          if (noHandStreakRef.current >= HAND_LOST_GRACE_FRAMES) {
            void finalizeCapture();
          }
        } else if (!showSuccessOverlay) {
          liveClassBufferRef.current = [];
          setPrediction((prev) => (prev.detected ? { ...prev, detected: false, gesture: "Menunggu tangan..." } : prev));
        }
      } else if (!handsDetected && isCapturingRef.current) {
        // Game selesai/overlay muncul di tengah capture → batalkan capture yang sedang berjalan
        rawBufferRef.current = [];
        isCapturingRef.current = false;
        noHandStreakRef.current = 0;
        liveClassBufferRef.current = [];
        setIsCapturing(false);
      }
    } catch (err) {
      console.error("Frame processing error:", err);
    }

    animIdRef.current = requestAnimationFrame(detectFrame);
  }, [isCameraOn, showSuccessOverlay, gameState.isFinished, activeQuestion, finalizeCapture, runLivePredict]);

  useEffect(() => {
    if (isCameraOn) {
      lastFrameTimeRef.current = performance.now();
      animIdRef.current = requestAnimationFrame(detectFrame);
    }
    return () => {
      if (animIdRef.current) cancelAnimationFrame(animIdRef.current);
    };
  }, [isCameraOn, detectFrame]);

  return (
    <main className="w-full px-4 sm:px-8 py-6 sm:py-8 max-w-7xl mx-auto space-y-6">
      <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl sm:text-3xl font-semibold text-ink flex items-center gap-2">
              <Gamepad2 className="h-7 w-7 text-teal" /> Tebak Kata
          </h1>
          <p className="text-ink-soft text-sm mt-1 max-w-prose">
            Uji kemampuan bahasa isyarat Anda! Peragakan kata yang diminta dan lihat skor serta tingkat keakuratannya.
          </p>
        </div>
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-teal-soft text-teal border border-teal/20">
            <Sparkles className="h-3.5 w-3.5" /> Model Full Class ({ACTIONS.length} Kata)
          </span>
        </div>
      </header>

      {!MODEL_READY && (
        <div className="rounded-xl border border-amber/40 bg-amber-soft px-4 py-3 text-sm text-ink">
          Model belum dikonfigurasi sepenuhnya. Fitur permainan ini membutuhkan model yang sudah dilatih.
        </div>
      )}

      {/* items-start: kolom kiri & kanan tidak lagi saling memaksa tinggi satu sama lain,
          jadi tidak ada ruang kosong dari efek grid "stretch" bawaan. */}
      <div className="grid grid-cols-1 lg:grid-cols-[1.6fr_1fr] gap-6 items-start">

        {/* LEFT COL: Game Area + Riwayat (dipindah ke sini agar mengisi tinggi kolom kiri) */}
        <section className="flex flex-col gap-4">
          <div className="relative w-full aspect-video rounded-2xl overflow-hidden bg-ink/90 border border-surface-border shadow-panel">
            {isLoadingModels && (
              <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 bg-ink/80 text-white">
                <div className="h-8 w-8 rounded-full border-2 border-amber border-t-transparent animate-spin" />
                <span className="text-sm font-medium">Memuat model...</span>
              </div>
            )}
            {initError && (
              <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-2 bg-warn/90 text-white px-6 text-center">
                <span className="font-semibold">Gagal memuat MediaPipe</span>
                <span className="text-sm">{initError}</span>
              </div>
            )}

            <video ref={videoRef} autoPlay playsInline muted className={`absolute inset-0 w-full h-full object-cover ${isMirrored ? "-scale-x-100" : ""}`} />
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover pointer-events-none" />

            {isCapturing && (
              <div className="absolute top-3 left-3 z-10 flex items-center gap-1.5 bg-warn/90 text-white text-xs font-medium px-2.5 py-1 rounded-full">
                <Circle className="h-2 w-2 fill-current animate-pulse" /> Merekam
              </div>
            )}

            {!isCameraOn && !isLoadingModels && !initError && (
              <div className="absolute inset-0 flex items-center justify-center bg-ink/95">
                <div className="text-center px-6">
                  <Camera className="h-10 w-10 text-amber mx-auto mb-3" />
                  <p className="text-white font-medium">Kamera belum aktif</p>
                </div>
              </div>
            )}

            {gameState.isFinished && (
               <div className="absolute inset-0 z-30 bg-ink/95 backdrop-blur-sm flex flex-col items-center justify-center text-white p-6 text-center animate-in fade-in duration-500">
                 <Trophy className="h-20 w-20 text-amber mb-4" />
                 <h2 className="text-3xl font-display font-bold mb-2">Permainan Selesai!</h2>
                 <p className="text-white/70 mb-4 max-w-md">Luar biasa! Anda telah menyelesaikan semua soal tebak kata.</p>
                 <div className="flex flex-wrap items-center justify-center gap-3 mb-6 text-sm">
                   <span className="bg-ok/20 border border-ok/30 text-ok px-3 py-1 rounded-full font-semibold">
                     ✓ {answeredQuestions.length} Terjawab
                   </span>
                   {skippedQuestions.length > 0 && (
                     <span className="bg-warn/20 border border-warn/30 text-warn px-3 py-1 rounded-full font-semibold">
                       ✕ {skippedQuestions.length} Dilewati
                     </span>
                   )}
                   <span className="bg-amber/20 border border-amber/30 text-amber px-3 py-1 rounded-full font-semibold">
                     ★ {totalScore} Poin
                   </span>
                 </div>
                 <div className="flex gap-4">
                    <button onClick={handleReset} className="px-6 py-2.5 bg-teal text-white rounded-lg font-medium hover:bg-teal/90 transition-colors">Ulangi dari Awal</button>
                    <button onClick={handleShuffleReset} className="px-6 py-2.5 bg-surface text-ink rounded-lg font-medium hover:bg-surface-border transition-colors">Acak Soal</button>
                 </div>
               </div>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2 bg-surface border border-surface-border rounded-xl px-3 py-2.5 shadow-panel">
            <button onClick={startCamera} disabled={isCameraOn || isLoadingModels || !!initError} className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium bg-teal text-white disabled:opacity-40 hover:bg-teal/90 transition-colors">
              <Camera className="h-4 w-4" /> Nyalakan kamera
            </button>
            <button onClick={stopCamera} disabled={!isCameraOn} className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium bg-warn/10 text-warn disabled:opacity-40 hover:bg-warn/20 transition-colors">
              <CameraOff className="h-4 w-4" /> Matikan
            </button>
            <button onClick={() => setIsMirrored((v) => !v)} className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-colors ${isMirrored ? "bg-teal-soft text-teal" : "bg-background text-ink-soft"}`}>
              <FlipHorizontal2 className="h-4 w-4" /> Cermin: {isMirrored ? "aktif" : "nonaktif"}
            </button>
          </div>

          {/* Riwayat Permainan — sebelumnya di bawah grid (full width), sekarang jadi
              bagian kolom kiri supaya langsung menyambung tanpa jarak kosong. */}
          <section className="bg-surface border border-surface-border rounded-2xl p-6 shadow-panel">
             <div className="flex items-center gap-2 mb-4 font-semibold text-ink">
                 <History className="h-5 w-5 text-teal" /> Riwayat Permainan
             </div>
             {history.length === 0 ? (
                 <div className="text-center py-8 text-ink-soft text-sm">
                     Belum ada riwayat permainan. Selesaikan soal atau reset untuk menyimpan progres.
                 </div>
             ) : (
                 <div className="overflow-x-auto">
                     <table className="w-full text-left text-sm text-ink border-collapse">
                         <thead>
                             <tr className="border-b border-surface-border text-ink-soft">
                                 <th className="py-3 px-4 font-medium">Waktu</th>
                                 <th className="py-3 px-4 font-medium">Soal Terjawab</th>
                                 <th className="py-3 px-4 font-medium">Akurasi Rata-rata</th>
                                 <th className="py-3 px-4 font-medium">Total Skor</th>
                             </tr>
                         </thead>
                         <tbody>
                             {history.map((h, hIdx) => (
                                 <tr key={`${h.id}-${hIdx}`} className="border-b border-surface-border/50 hover:bg-background/50 transition-colors">
                                     <td className="py-3 px-4">{h.date}</td>
                                     <td className="py-3 px-4">
                                         <div className="flex items-center gap-1.5 flex-wrap">
                                             <span className="inline-flex items-center justify-center bg-teal-soft text-teal px-2 py-0.5 rounded-full font-medium">
                                                 {h.answeredCount} / {h.totalQuestions}
                                             </span>
                                             {Boolean(h.skippedCount) && (
                                                 <span className="inline-flex items-center justify-center bg-warn-soft text-warn px-2 py-0.5 rounded-full text-xs font-semibold">
                                                     {h.skippedCount} dilewati
                                                 </span>
                                             )}
                                         </div>
                                     </td>
                                     <td className="py-3 px-4 font-medium text-ok">{h.accuracy}%</td>
                                     <td className="py-3 px-4 font-bold">{h.score}</td>
                                 </tr>
                             ))}
                         </tbody>
                     </table>
                 </div>
             )}
          </section>
        </section>

        {/* RIGHT COL: Questions List & Controls */}
        <aside className="flex flex-col gap-4">
           {/* BIG CARD: Current Question */}
           <div className="bg-surface border border-surface-border rounded-2xl p-6 shadow-panel text-center flex flex-col items-center justify-center min-h-48 relative overflow-hidden">
            {showSuccessOverlay ? (
              <div className="absolute inset-0 bg-ok flex flex-col items-center justify-center text-white animate-in zoom-in duration-300">
                <CheckCircle2 className="h-14 w-14 mb-2" />
                <h2 className="text-2xl font-display font-bold">BENAR!</h2>
                <div className="text-sm font-medium opacity-90 mt-1">
                   +{gameState.questions[gameState.currentIndex - 1]?.score} Poin
                </div>
              </div>
            ) : (
                <>
                    <div className="text-ink-soft text-sm font-medium mb-1">
                      {gameState.isFinished ? "Permainan Selesai" : "Tebak isyarat untuk kata:"}
                    </div>
                    <div className="text-4xl sm:text-5xl font-display font-bold text-ink tracking-tight mb-3">
                        {gameState.isFinished ? "SELESAI" : (activeQuestion?.word.toUpperCase() || "SELESAI")}
                    </div>
                    {!gameState.isFinished && activeQuestion && (
                      <button
                        type="button"
                        onClick={handleSkip}
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold bg-warn-soft text-warn hover:bg-warn hover:text-white border border-warn/30 transition-all shadow-sm active:scale-95 cursor-pointer"
                        title="Lewati ke kata urutan selanjutnya"
                      >
                        <SkipForward className="h-4 w-4" /> Lewati Kata (Skip)
                      </button>
                    )}
                </>
            )}
          </div>

          {/* PREDICTION MODEL CARD — menampilkan apa yang model tebak (kelas + akurasi)
              selain kata target yang harus diperagakan. */}
          <div className="bg-surface border border-surface-border rounded-2xl p-4 shadow-panel">
            <div className="flex items-center gap-2 text-ink-soft text-xs font-medium mb-2">
              <Radar className="h-4 w-4 text-teal" /> Prediksi Model
            </div>
            {prediction.detected && prediction.gesture !== "Merekam gerakan..." ? (
              activeQuestion ? (
                <div className={`rounded-xl border px-3 py-2.5 ${predictionMatches ? "bg-ok-soft/40 border-ok/25" : "bg-warn-soft border-warn/25"}`}>
                  <div className="flex items-baseline justify-between gap-2">
                    <span className={`text-xl font-display font-semibold ${predictionMatches ? "text-ok" : "text-warn"}`}>
                      {prediction.gesture}
                    </span>
                    <span className={`text-sm font-bold ${predictionMatches ? "text-ok" : "text-warn"}`}>
                      {Math.round(prediction.confidence * 100)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-2 mt-1">
                    <div className="text-xs text-ink-soft">
                      {prediction.isLive
                        ? "Prediksi sementara saat gesture berlangsung"
                        : predictionMatches
                          ? "Tebakan model sesuai kata target"
                          : `Model menebak kata lain dari target "${activeQuestion.word}"`}
                    </div>
                    {prediction.isLive && (
                      <span className="inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-background text-teal border border-teal/25 shrink-0">
                        <Circle className="h-1.5 w-1.5 fill-current animate-pulse" /> LIVE
                      </span>
                    )}
                  </div>
                </div>
              ) : (
                <div className="text-sm text-ink-soft">
                  {prediction.gesture} · {Math.round(prediction.confidence * 100)}%
                </div>
              )
            ) : (
              <div className="text-sm text-ink-soft">
                {isCapturing
                  ? "Merekam gerakan — terus peragakan isyarat..."
                  : prediction.gesture === "Menunggu..."
                    ? "Belum ada prediksi. Peragakan isyarat untuk kata target."
                    : prediction.gesture}
              </div>
            )}
          </div>

          {/* SCORE & ACCURACY CARDS */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-surface border border-surface-border rounded-2xl p-4 shadow-panel flex items-center gap-3">
                <div className="h-10 w-10 rounded-full bg-amber-soft text-amber flex items-center justify-center shrink-0">
                    <Trophy className="h-5 w-5" />
                </div>
                <div className="min-w-0">
                    <div className="text-xs font-medium text-ink-soft truncate">Skor Anda</div>
                    <div className="text-xl font-bold text-ink truncate">{totalScore}</div>
                </div>
            </div>
            <div className="bg-surface border border-surface-border rounded-2xl p-4 shadow-panel flex items-center gap-3">
                <div className="h-10 w-10 rounded-full bg-teal-soft text-teal flex items-center justify-center shrink-0">
                    <Target className="h-5 w-5" />
                </div>
                <div className="min-w-0">
                    <div className="text-xs font-medium text-ink-soft truncate">Akurasi</div>
                    <div className="text-xl font-bold text-teal truncate">{avgAccuracy}%</div>
                </div>
            </div>
          </div>

           {/* QUESTIONS LIST */}
           <div className="bg-surface border border-surface-border rounded-2xl shadow-panel flex flex-col">
               <div className="p-4 border-b border-surface-border flex items-center justify-between bg-background">
                  <div className="flex items-center gap-2 font-medium text-ink">
                      <ListChecks className="h-5 w-5 text-teal" /> Daftar Soal
                  </div>
                  <div className="flex items-center gap-2">
                     {skippedQuestions.length > 0 && (
                       <span className="text-xs font-semibold text-warn bg-warn-soft border border-warn/25 px-2 py-0.5 rounded-full">
                         {skippedQuestions.length} dilewati
                       </span>
                     )}
                     <span className="text-sm font-semibold text-ink-soft bg-surface border border-surface-border px-2 py-0.5 rounded-full">
                        {answeredQuestions.length} / {gameState.questions.length}
                     </span>
                  </div>
               </div>

               <div className="max-h-[26rem] overflow-y-auto p-2 space-y-1">
                   {gameState.questions.map((q, idx) => {
                       const isCurrent = idx === gameState.currentIndex && !gameState.isFinished;
                       return (
                           <div key={q.word + idx}
                                className={`flex items-center justify-between p-3 rounded-xl border transition-colors ${
                                    q.isAnswered
                                    ? "bg-ok-soft/30 border-ok/20"
                                    : q.isSkipped
                                        ? "bg-warn-soft/40 border-warn/30"
                                        : isCurrent
                                            ? "bg-teal-soft/50 border-teal shadow-sm"
                                            : "bg-surface border-transparent"
                                }`}
                           >
                               <div className="flex items-center gap-3">
                                   <div className={`flex items-center justify-center h-6 w-6 rounded-full text-xs font-bold ${
                                       q.isAnswered
                                         ? "bg-ok text-white"
                                         : q.isSkipped
                                           ? "bg-warn text-white"
                                           : isCurrent
                                             ? "bg-teal text-white"
                                             : "bg-background text-ink-soft"
                                   }`}>
                                       {idx + 1}
                                   </div>
                                   <span className={`font-medium ${
                                       q.isAnswered
                                         ? "text-ok"
                                         : q.isSkipped
                                           ? "text-warn font-semibold"
                                           : isCurrent
                                             ? "text-teal font-semibold"
                                             : "text-ink"
                                   }`}>
                                       {q.word}
                                   </span>
                               </div>
                               {q.isAnswered && (
                                   <div className="flex flex-col items-end">
                                      <span className="text-xs font-bold text-ok">{q.accuracy}%</span>
                                      <span className="text-[10px] text-ok/70">{q.score} pt</span>
                                   </div>
                               )}
                               {q.isSkipped && (
                                   <div className="flex flex-col items-end">
                                      <span className="text-xs font-semibold text-warn bg-warn/10 px-2 py-0.5 rounded-full border border-warn/20">
                                        Dilewati
                                      </span>
                                   </div>
                               )}
                               {isCurrent && (
                                   <button
                                     type="button"
                                     onClick={handleSkip}
                                     className="text-xs font-semibold px-2 py-1 rounded-lg bg-warn-soft text-warn hover:bg-warn hover:text-white border border-warn/30 transition-all flex items-center gap-1 active:scale-95 cursor-pointer"
                                     title="Lewati kata ini"
                                   >
                                     Skip <SkipForward className="h-3 w-3" />
                                   </button>
                               )}
                           </div>
                       );
                   })}
               </div>

               <div className="p-4 border-t border-surface-border bg-background grid grid-cols-2 gap-2">
                   <button onClick={handleReset} className="flex items-center justify-center gap-2 py-2 rounded-lg bg-surface border border-surface-border text-ink hover:bg-teal-soft hover:text-teal hover:border-teal transition-colors text-sm font-medium">
                       <RotateCcw className="h-4 w-4" /> Reset
                   </button>
                   <button onClick={handleShuffleReset} className="flex items-center justify-center gap-2 py-2 rounded-lg bg-surface border border-surface-border text-ink hover:bg-teal-soft hover:text-teal hover:border-teal transition-colors text-sm font-medium">
                       <Shuffle className="h-4 w-4" /> Acak
                   </button>
               </div>
           </div>
        </aside>
      </div>
    </main>
  );
}
