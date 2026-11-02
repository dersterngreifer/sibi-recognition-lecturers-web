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
import { ACTIONS, SEQUENCE_LENGTH, STABILITY_FRAMES, THRESHOLD } from "@/config/modelConfig";
import {
  Camera, CameraOff, FlipHorizontal2, MessageSquareText,
  Delete, Sparkles, CheckCircle2, XCircle, Loader2, RotateCcw
} from "lucide-react";

const MODEL_READY = Boolean(SEQUENCE_LENGTH && STABILITY_FRAMES && THRESHOLD !== null);

type VerificationResult = {
  isValid: boolean;
  message: string;
  suggestion?: string;
};

export default function UjiKalimatPage() {
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

  const [prediction, setPrediction] = useState({ gesture: "Menunggu...", confidence: 0, detected: false });

  // --- Sentence Builder State ---
  const [sentence, setSentence] = useState<string[]>([]);
  const [lastAddedWord, setLastAddedWord] = useState<string | null>(null);
  const noHandsFramesRef = useRef(0);

  // --- AI Verification State ---
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<VerificationResult | null>(null);

  // Initialize MediaPipe & Model
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

  const isMirroredRef = useRef(isMirrored);
  const sequenceRef = useRef<Float32Array[]>([]);
  const predictionBufferRef = useRef<number[]>([]);
  const lastFrameTimeRef = useRef(0);
  const frameCountRef = useRef(0);
  const animIdRef = useRef<number | null>(null);
  const lastUiUpdateRef = useRef(0);

  useEffect(() => { isMirroredRef.current = isMirrored; }, [isMirrored]);

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
      setVerificationResult(null);
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

      if (handsDetected && result && MODEL_READY && !isVerifying) {
        noHandsFramesRef.current = 0;
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
                const guessedWord = ACTIONS[idx];
                
                // Only add if it's a different word from the last added one
                setLastAddedWord(prev => {
                    if (prev !== guessedWord) {
                        setSentence(s => [...s, guessedWord]);
                        return guessedWord;
                    }
                    return prev;
                });
              }
            }

            setPrediction({
              gesture: ACTIONS[idx],
              confidence: res.confidence,
              detected: true,
            });
          }
        }
      } else if (!handsDetected) {
        sequenceRef.current = [];
        predictionBufferRef.current = [];
        setPrediction((prev) => (prev.detected ? { ...prev, detected: false, gesture: "Menunggu tangan..." } : prev));
        
        // If no hands detected for roughly 1 second (assuming ~30fps, 30 frames), reset lastAddedWord
        // so they can sign the same word again
        noHandsFramesRef.current++;
        if (noHandsFramesRef.current > 30) {
            setLastAddedWord(null);
        }
      }
    } catch (err) {
      console.error("Frame processing error:", err);
    }

    animIdRef.current = requestAnimationFrame(detectFrame);
  }, [isCameraOn, isVerifying]);

  useEffect(() => {
    if (isCameraOn) {
      lastFrameTimeRef.current = performance.now();
      animIdRef.current = requestAnimationFrame(detectFrame);
    }
    return () => {
      if (animIdRef.current) cancelAnimationFrame(animIdRef.current);
    };
  }, [isCameraOn, detectFrame]);


  // --- Sentence Actions ---
  const handleRemoveLastWord = () => {
      setSentence(prev => prev.slice(0, -1));
      setLastAddedWord(null);
      setVerificationResult(null);
  };

  const handleClearSentence = () => {
      setSentence([]);
      setLastAddedWord(null);
      setVerificationResult(null);
  };

  const handleVerifyAI = async () => {
      if (sentence.length === 0) return;
      
      setIsVerifying(true);
      setVerificationResult(null);

      // MOCK AI LLM DELAY
      await new Promise(r => setTimeout(r, 2500));

      // MOCK AI LOGIC (In production, replace with actual fetch to LLM endpoint)
      // e.g. await fetch('/api/verify-spok', { method: 'POST', body: JSON.stringify({ sentence }) })
      
      let isValid = false;
      let message = "";
      
      if (sentence.length < 2) {
          message = "Kalimat terlalu singkat. Minimal gunakan gabungan Subjek dan Predikat.";
      } else {
          // Dummy mock: 70% chance to be considered valid for demonstration
          isValid = Math.random() > 0.3;
          if (isValid) {
              message = "Struktur kalimat sudah baik dan beraturan sesuai dengan prinsip penyusunan kalimat dasar.";
          } else {
              message = "Struktur kalimat kurang tepat. Coba perhatikan kembali urutan Subjek, Predikat, dan Objek agar lebih jelas.";
          }
      }

      setVerificationResult({ isValid, message });
      setIsVerifying(false);
  };

  return (
    <main className="w-full px-4 sm:px-8 py-6 sm:py-8 max-w-7xl mx-auto space-y-6">
      <header>
        <h1 className="text-2xl sm:text-3xl font-semibold text-ink flex items-center gap-2">
            <MessageSquareText className="h-7 w-7 text-teal" /> Uji Kalimat (AI)
        </h1>
        <p className="text-ink-soft text-sm mt-1 max-w-prose">
          Susun kata secara bebas melalui bahasa isyarat untuk membentuk kalimat. 
          Model AI kami akan mengevaluasi apakah struktur kalimat Anda sudah tepat.
        </p>
      </header>

      {!MODEL_READY && (
        <div className="rounded-xl border border-amber/40 bg-amber-soft px-4 py-3 text-sm text-ink">
          Model belum dikonfigurasi sepenuhnya. Fitur ini membutuhkan model yang sudah dilatih.
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[1.6fr_1fr] gap-6">
        
        {/* LEFT COL: Camera */}
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

            {!isCameraOn && !isLoadingModels && !initError && (
              <div className="absolute inset-0 flex items-center justify-center bg-ink/95">
                <div className="text-center px-6">
                  <Camera className="h-10 w-10 text-amber mx-auto mb-3" />
                  <p className="text-white font-medium">Kamera belum aktif</p>
                </div>
              </div>
            )}

            {isCameraOn && (
              <div className="absolute top-3 left-3 z-20 flex items-center gap-2 bg-ink/70 backdrop-blur px-3 py-1.5 rounded-lg text-white text-xs font-medium">
                <span className="h-1.5 w-1.5 rounded-full bg-ok" />
                {stats.fps} fps
              </div>
            )}

            {isCameraOn && prediction.detected && !isVerifying && (
               <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2 bg-ink/80 backdrop-blur-md px-6 py-2 rounded-full border border-white/10 shadow-xl">
                  <span className="text-white font-display font-medium text-lg">{prediction.gesture}</span>
                  <span className="text-teal font-bold text-sm bg-teal-soft/20 px-2 py-0.5 rounded-md">
                     {Math.round(prediction.confidence * 100)}%
                  </span>
               </div>
            )}
            
            {isVerifying && (
                <div className="absolute inset-0 z-30 bg-ink/80 backdrop-blur-sm flex flex-col items-center justify-center text-white">
                    <Loader2 className="h-12 w-12 text-teal animate-spin mb-3" />
                    <span className="font-medium text-lg">AI sedang menganalisis kalimat...</span>
                </div>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2 bg-surface border border-surface-border rounded-xl px-3 py-2.5 shadow-panel">
            <button onClick={startCamera} disabled={isCameraOn || isLoadingModels || !!initError} className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium bg-teal text-white disabled:opacity-40 hover:bg-teal/90 transition-colors">
              <Camera className="h-4 w-4" /> Nyalakan
            </button>
            <button onClick={stopCamera} disabled={!isCameraOn} className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium bg-warn/10 text-warn disabled:opacity-40 hover:bg-warn/20 transition-colors">
              <CameraOff className="h-4 w-4" /> Matikan
            </button>
            <button onClick={() => setIsMirrored((v) => !v)} className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-colors ${isMirrored ? "bg-teal-soft text-teal" : "bg-background text-ink-soft"}`}>
              <FlipHorizontal2 className="h-4 w-4" /> Cermin: {isMirrored ? "aktif" : "nonaktif"}
            </button>
          </div>
        </section>

        {/* RIGHT COL: Sentence Builder & AI Feedback */}
        <aside className="flex flex-col gap-4">
           
           {/* SENTENCE BUILDER */}
           <div className="bg-surface border border-surface-border rounded-2xl p-5 shadow-panel flex flex-col min-h-60">
               <div className="flex items-center justify-between mb-4">
                   <div className="text-sm font-medium text-ink-soft">Kalimat yang disusun:</div>
                   <div className="flex gap-2">
                       <button 
                           onClick={handleRemoveLastWord} 
                           disabled={sentence.length === 0 || isVerifying}
                           className="p-1.5 text-ink-soft hover:text-warn hover:bg-warn-soft rounded-md disabled:opacity-30 transition-colors"
                           title="Hapus kata terakhir"
                       >
                           <Delete className="h-5 w-5" />
                       </button>
                       <button 
                           onClick={handleClearSentence} 
                           disabled={sentence.length === 0 || isVerifying}
                           className="p-1.5 text-ink-soft hover:text-warn hover:bg-warn-soft rounded-md disabled:opacity-30 transition-colors"
                           title="Hapus semua kata"
                       >
                           <RotateCcw className="h-5 w-5" />
                       </button>
                   </div>
               </div>
               
               <div className="flex-1 bg-background border border-surface-border rounded-xl p-4 flex flex-wrap content-start gap-2 min-h-32">
                   {sentence.length === 0 ? (
                       <span className="text-ink-soft/60 italic text-sm m-auto">Belum ada kata yang terdeteksi. Silakan mulai berisyarat.</span>
                   ) : (
                       sentence.map((word, idx) => (
                           <span key={`${word}-${idx}`} className="inline-flex items-center px-3 py-1.5 rounded-lg bg-teal text-white font-medium text-lg shadow-sm animate-in zoom-in duration-200">
                               {word}
                           </span>
                       ))
                   )}
               </div>

               <button 
                  onClick={handleVerifyAI}
                  disabled={sentence.length === 0 || isVerifying}
                  className="mt-4 w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold bg-ink text-white hover:bg-ink/90 transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-md"
               >
                   <Sparkles className="h-5 w-5 text-amber" /> Verifikasi dengan AI
               </button>
           </div>

           {/* AI FEEDBACK */}
           <div className={`bg-surface border border-surface-border rounded-2xl p-5 shadow-panel flex-1 transition-all duration-300 ${verificationResult ? 'border-teal' : ''}`}>
               <div className="flex items-center gap-2 mb-3">
                   <Sparkles className="h-5 w-5 text-teal" />
                   <h3 className="font-semibold text-ink">Hasil Analisis AI</h3>
               </div>
               
               {!verificationResult ? (
                   <p className="text-sm text-ink-soft italic">Tekan tombol Verifikasi untuk melihat hasil analisis struktur kalimat (SPOK) Anda.</p>
               ) : (
                   <div className="space-y-3 animate-in fade-in slide-in-from-bottom-2 duration-300">
                       <div className={`flex items-start gap-3 p-3 rounded-xl border ${verificationResult.isValid ? 'bg-ok-soft border-ok/30 text-ok' : 'bg-warn-soft border-warn/30 text-warn'}`}>
                           {verificationResult.isValid ? (
                               <CheckCircle2 className="h-6 w-6 shrink-0 mt-0.5" />
                           ) : (
                               <XCircle className="h-6 w-6 shrink-0 mt-0.5" />
                           )}
                           <div>
                               <div className="font-semibold">{verificationResult.isValid ? "Struktur Valid" : "Perlu Perbaikan"}</div>
                               <p className="text-sm mt-1 text-ink/80 leading-relaxed">{verificationResult.message}</p>
                           </div>
                       </div>
                   </div>
               )}
           </div>

        </aside>
      </div>
    </main>
  );
}
