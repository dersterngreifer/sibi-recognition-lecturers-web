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
  ACTIONS,
  SEQUENCE_LENGTH,
  STABILITY_FRAMES,
  THRESHOLD,
} from "@/config/modelConfig";
import {
  Camera,
  CameraOff,
  FlipHorizontal2,
  Radar,
  History,
  Gauge,
} from "lucide-react";

type Prediction = {
  gesture: string;
  confidence: number;
  allProbabilities: number[];
  detected: boolean;
};

type HistoryEntry = { word: string; time: string; confidence: number };

const MODEL_READY = Boolean(SEQUENCE_LENGTH && STABILITY_FRAMES && THRESHOLD !== null);

export default function BerandaPage() {
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

  const [prediction, setPrediction] = useState<Prediction>({
    gesture: "Menunggu...",
    confidence: 0,
    allProbabilities: Array(ACTIONS.length).fill(0),
    detected: false,
  });
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  const isMirroredRef = useRef(isMirrored);
  const sequenceRef = useRef<Float32Array[]>([]);
  const predictionBufferRef = useRef<number[]>([]);
  const lastFrameTimeRef = useRef(0);
  const frameCountRef = useRef(0);
  const animIdRef = useRef<number | null>(null);
  const lastUiUpdateRef = useRef(0);

  useEffect(() => {
    isMirroredRef.current = isMirrored;
  }, [isMirrored]);

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
        const modelOk = await loadModel();
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
    };
  }, []);

  const startCamera = async () => {
    try {
      if (!isModelLoaded()) loadModel().catch(() => {});
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      });
      setStream(mediaStream);
      if (videoRef.current) videoRef.current.srcObject = mediaStream;
      setIsCameraOn(true);
      sequenceRef.current = [];
      predictionBufferRef.current = [];
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
      sequenceRef.current = [];
    }
  };

  useEffect(() => () => stream?.getTracks().forEach((t) => t.stop()), [stream]);

  const detectFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isCameraOn || videoRef.current.videoWidth === 0) {
      animIdRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    try {
      const now = performance.now();
      const delta = now - lastFrameTimeRef.current;
      lastFrameTimeRef.current = now;
      frameCountRef.current++;

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

      if (handsDetected && result && MODEL_READY) {
        const kp = extractKeypoints(result, isMirroredRef.current);
        sequenceRef.current.push(kp);
        sequenceRef.current = sequenceRef.current.slice(-(SEQUENCE_LENGTH as number));

        if (sequenceRef.current.length === SEQUENCE_LENGTH && frameCountRef.current % 4 === 0) {
          const res = await predictGesture(sequenceRef.current);

          if (res.predictions.length > 0) {
            const idx = res.predictions[0];
            predictionBufferRef.current.push(idx);
            predictionBufferRef.current = predictionBufferRef.current.slice(-(STABILITY_FRAMES as number));

            if (predictionBufferRef.current.length === STABILITY_FRAMES) {
              const allSame = predictionBufferRef.current.every((p) => p === predictionBufferRef.current[0]);
              if (allSame && res.confidence > (THRESHOLD as number)) {
                setHistory((prev) => {
                  const word = ACTIONS[idx];
                  const time = new Date().toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
                  if (prev[0]?.word === word) return prev;
                  return [{ word, time, confidence: Math.round(res.confidence * 100) }, ...prev].slice(0, 20);
                });
              }
            }

            setPrediction({
              gesture: ACTIONS[idx],
              confidence: res.confidence,
              allProbabilities: res.allProbabilities,
              detected: true,
            });
          }
        }
      } else if (!handsDetected) {
        sequenceRef.current = [];
        predictionBufferRef.current = [];
        setPrediction((prev) => (prev.detected ? { ...prev, detected: false, gesture: "Menunggu tangan..." } : prev));
      }
    } catch (err) {
      console.error("Frame processing error:", err);
    }

    animIdRef.current = requestAnimationFrame(detectFrame);
  }, [isCameraOn]);

  useEffect(() => {
    if (isCameraOn) {
      lastFrameTimeRef.current = performance.now();
      animIdRef.current = requestAnimationFrame(detectFrame);
    }
    return () => {
      if (animIdRef.current) cancelAnimationFrame(animIdRef.current);
    };
  }, [isCameraOn, detectFrame]);

  const topProbabilities = prediction.allProbabilities
    .map((p, i) => ({ word: ACTIONS[i], p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 5);

  return (
    <main className="w-full px-4 sm:px-8 py-6 sm:py-8 max-w-6xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl sm:text-3xl font-semibold text-ink">Susun Kata</h1>
        <p className="text-ink-soft text-sm mt-1 max-w-prose">
          Nyalakan kamera dan tunjukkan gerakan isyarat SIBI untuk menyusun kata secara langsung.
        </p>
      </header>

      {!MODEL_READY && (
        <div className="mb-6 rounded-xl border border-amber/40 bg-amber-soft px-4 py-3 text-sm text-ink">
          Model belum dikonfigurasi sepenuhnya (lihat <code className="font-mono text-xs">src/config/modelConfig.ts</code>).
          Kamera & kerangka tetap bisa diuji, tapi prediksi kata belum aktif.
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[1.6fr_1fr] gap-5">
        {/* Camera column */}
        <section className="flex flex-col gap-3">
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

            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`absolute inset-0 w-full h-full object-cover ${isMirrored ? "-scale-x-100" : ""}`}
            />
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover pointer-events-none" />

            {!isCameraOn && !isLoadingModels && !initError && (
              <div className="absolute inset-0 flex items-center justify-center bg-ink/95">
                <div className="text-center px-6">
                  <Camera className="h-10 w-10 text-amber mx-auto mb-3" />
                  <p className="text-white font-medium">Kamera belum aktif</p>
                  <p className="text-white/60 text-sm mt-1">Tekan &ldquo;Nyalakan kamera&rdquo; untuk mulai</p>
                </div>
              </div>
            )}

            {isCameraOn && (
              <div className="absolute top-3 left-3 z-20 flex items-center gap-2 bg-ink/70 backdrop-blur px-3 py-1.5 rounded-lg text-white text-xs font-medium">
                <span className="h-1.5 w-1.5 rounded-full bg-ok" />
                {stats.fps} fps
              </div>
            )}

            {isCameraOn && (
              <div className="absolute top-3 right-3 z-20 bg-ink/70 backdrop-blur px-3 py-1.5 rounded-lg text-white text-xs font-medium">
                {distance > 0 ? `${distance} cm dari kamera` : "Posisi badan tidak terdeteksi"}
              </div>
            )}
          </div>

          {/* Toolbar under video — distinct from reference's floating overlay buttons */}
          <div className="flex flex-wrap items-center gap-2 bg-surface border border-surface-border rounded-xl px-3 py-2.5 shadow-panel">
            <button
              onClick={startCamera}
              disabled={isCameraOn || isLoadingModels || !!initError}
              className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium bg-teal text-white disabled:opacity-40 disabled:cursor-not-allowed hover:bg-teal/90 transition-colors"
            >
              <Camera className="h-4 w-4" /> Nyalakan kamera
            </button>
            <button
              onClick={stopCamera}
              disabled={!isCameraOn}
              className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium bg-warn/10 text-warn disabled:opacity-40 disabled:cursor-not-allowed hover:bg-warn/20 transition-colors"
            >
              <CameraOff className="h-4 w-4" /> Matikan
            </button>
            <button
              onClick={() => setIsMirrored((v) => !v)}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-colors ${
                isMirrored ? "bg-teal-soft text-teal" : "bg-background text-ink-soft"
              }`}
            >
              <FlipHorizontal2 className="h-4 w-4" /> Cermin: {isMirrored ? "aktif" : "nonaktif"}
            </button>
          </div>
        </section>

        {/* Result column */}
        <section className="flex flex-col gap-4">
          <div className="bg-surface border border-surface-border rounded-2xl p-4 shadow-panel">
            <div className="flex items-center gap-2 mb-3 text-ink-soft text-xs font-medium">
              <Radar className="h-4 w-4" /> Hasil saat ini
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-xl font-display font-semibold text-ink">{prediction.gesture}</span>
              {prediction.detected && (
                <span className="text-sm font-semibold text-teal">{Math.round(prediction.confidence * 100)}%</span>
              )}
            </div>
          </div>

          <div className="bg-surface border border-surface-border rounded-2xl p-4 shadow-panel">
            <div className="flex items-center gap-2 mb-3 text-ink-soft text-xs font-medium">
              <Gauge className="h-4 w-4" /> Probabilitas teratas
            </div>
            <div className="flex flex-col gap-2">
              {topProbabilities.map(({ word, p }) => (
                <div key={word} className="relative overflow-hidden rounded-lg bg-background">
                  <div
                    className="absolute inset-y-0 left-0 bg-teal-soft transition-all"
                    style={{ width: `${Math.round(p * 100)}%` }}
                  />
                  <div className="relative flex items-center justify-between px-3 py-1.5 text-sm">
                    <span className="text-ink font-medium">{word}</span>
                    <span className="text-ink-soft">{Math.round(p * 100)}%</span>
                  </div>
                </div>
              ))}
              {topProbabilities.every((t) => t.p === 0) && (
                <p className="text-sm text-ink-soft">Belum ada data probabilitas.</p>
              )}
            </div>
          </div>

          <div className="bg-surface border border-surface-border rounded-2xl p-4 shadow-panel flex-1 min-h-40">
            <div className="flex items-center gap-2 mb-3 text-ink-soft text-xs font-medium">
              <History className="h-4 w-4" /> Riwayat kata terdeteksi
            </div>
            {history.length === 0 ? (
              <p className="text-sm text-ink-soft">Kata yang berhasil dikenali akan muncul di sini.</p>
            ) : (
              <ul className="flex flex-col gap-1.5 max-h-72 overflow-y-auto pr-1">
                {history.map((h, i) => (
                  <li
                    key={`${h.word}-${h.time}-${i}`}
                    className="flex items-center justify-between text-sm px-3 py-1.5 rounded-lg bg-background"
                  >
                    <span className="font-medium text-ink">{h.word}</span>
                    <span className="text-ink-soft text-xs">{h.time} · {h.confidence}%</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
