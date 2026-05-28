"use client";

import { useState, useRef, useCallback } from "react";

const API = "http://localhost:8000";

type Status = "idle" | "uploaded" | "queued" | "processing" | "complete" | "failed";
type Mode = "bw" | "full";

interface Results {
  bw?: string;
  color?: string;
}

export default function Home() {
  const [inputPreview, setInputPreview] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [stage, setStage] = useState<string>("");
  const [progress, setProgress] = useState<number>(0);
  const [mode, setMode] = useState<Mode>("full");
  const [colorTheme, setColorTheme] = useState<string>("");
  const [results, setResults] = useState<Results>({});
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const onFileSelect = useCallback(async (file: File) => {
    setError(null);
    setResults({});
    setProgress(0);
    setStage("");

    const preview = URL.createObjectURL(file);
    setInputPreview(preview);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API}/api/upload`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }
      const data = await res.json();
      setJobId(data.job_id);
      setStatus("uploaded");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setStatus("failed");
    }
  }, []);

  const startGeneration = useCallback(async () => {
    if (!jobId) return;
    setError(null);
    setStatus("queued");
    setProgress(0);

    try {
      const res = await fetch(`${API}/api/generate/${jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, color_theme: colorTheme || undefined }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Generation failed");
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Generation failed");
      setStatus("failed");
      return;
    }

    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API}/api/status/${jobId}`);
        const data = await res.json();
        setStatus(data.status);
        setStage(data.stage || "");
        setProgress(data.progress || 0);

        if (data.status === "complete") {
          clearInterval(pollRef.current!);
          const resultRes = await fetch(`${API}/api/result/${jobId}`);
          const resultData = await resultRes.json();
          setResults(resultData);
        } else if (data.status === "failed") {
          clearInterval(pollRef.current!);
          setError(data.error || "Generation failed");
        }
      } catch {
        // poll continues
      }
    }, 2000);
  }, [jobId, mode, colorTheme]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) onFileSelect(file);
    },
    [onFileSelect]
  );

  const download = (variant: "bw" | "color") => {
    if (!jobId) return;
    window.open(`${API}/api/download/${jobId}/${variant}`, "_blank");
  };

  const reset = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    setInputPreview(null);
    setJobId(null);
    setStatus("idle");
    setStage("");
    setProgress(0);
    setResults({});
    setError(null);
  };

  const isProcessing = status === "queued" || status === "processing";

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white">GEGEAI 2.0</h1>
          <p className="text-xs text-zinc-500 mt-0.5">Gege Akutami Style Transfer</p>
        </div>
        <div className="flex gap-1 bg-zinc-900 rounded-lg p-1">
          <button
            onClick={() => setMode("bw")}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
              mode === "bw"
                ? "bg-zinc-700 text-white"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            B&W Only
          </button>
          <button
            onClick={() => setMode("full")}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
              mode === "full"
                ? "bg-zinc-700 text-white"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            B&W + Color
          </button>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 flex gap-6 p-6 max-w-6xl mx-auto w-full">
        {/* Left panel */}
        <div className="w-80 flex-shrink-0 flex flex-col gap-4">
          {/* Dropzone */}
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            className={`relative aspect-square rounded-xl border-2 border-dashed cursor-pointer flex flex-col items-center justify-center transition-colors overflow-hidden ${
              dragging
                ? "border-zinc-400 bg-zinc-800"
                : "border-zinc-700 bg-zinc-900 hover:border-zinc-500 hover:bg-zinc-800"
            }`}
          >
            {inputPreview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={inputPreview} alt="Input" className="absolute inset-0 w-full h-full object-cover" />
            ) : (
              <div className="text-center p-4">
                <svg className="w-10 h-10 text-zinc-600 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p className="text-sm text-zinc-400">Drop image here</p>
                <p className="text-xs text-zinc-600 mt-1">JPG, PNG, WebP — max 50MB</p>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp"
              className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) onFileSelect(f); }}
            />
          </div>

          {/* Color theme input (full mode only) */}
          {mode === "full" && (
            <div>
              <label className="block text-xs text-zinc-500 mb-1.5">Color Theme Hint (optional)</label>
              <input
                type="text"
                value={colorTheme}
                onChange={(e) => setColorTheme(e.target.value)}
                placeholder="e.g. warm sunset tones"
                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500"
              />
            </div>
          )}

          {/* Controls */}
          <div className="flex flex-col gap-2">
            {status === "uploaded" || (status === "complete" && jobId) ? (
              <>
                <button
                  onClick={startGeneration}
                  disabled={isProcessing}
                  className="w-full py-2.5 rounded-lg bg-white text-zinc-950 font-semibold text-sm hover:bg-zinc-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Generate
                </button>
                <button
                  onClick={reset}
                  className="w-full py-2 rounded-lg border border-zinc-700 text-zinc-400 text-sm hover:border-zinc-500 hover:text-zinc-200 transition-colors"
                >
                  Reset
                </button>
              </>
            ) : status === "idle" ? (
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full py-2.5 rounded-lg bg-zinc-800 text-zinc-300 text-sm hover:bg-zinc-700 transition-colors"
              >
                Select Image
              </button>
            ) : null}
          </div>

          {/* Progress */}
          {isProcessing && (
            <div>
              <div className="flex justify-between text-xs text-zinc-500 mb-1.5">
                <span>{stage || "Processing..."}</span>
                <span>{progress}%</span>
              </div>
              <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-white rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="rounded-lg bg-red-950 border border-red-800 px-3 py-2 text-xs text-red-300">
              {error}
            </div>
          )}

          {/* Status badge */}
          {status !== "idle" && (
            <div className={`text-xs px-2 py-1 rounded text-center font-mono ${
              status === "complete" ? "bg-green-950 text-green-400" :
              status === "failed" ? "bg-red-950 text-red-400" :
              "bg-zinc-800 text-zinc-400"
            }`}>
              {status.toUpperCase()}
            </div>
          )}
        </div>

        {/* Right panel — results */}
        <div className="flex-1 flex flex-col gap-6">
          {status === "complete" && (results.bw || results.color) ? (
            <>
              <div className={`grid gap-6 ${results.bw && results.color ? "grid-cols-2" : "grid-cols-1"}`}>
                {results.bw && (
                  <ResultCard
                    label="B&W — Gege Manga Style"
                    src={`data:image/png;base64,${results.bw}`}
                    onDownload={() => download("bw")}
                  />
                )}
                {results.color && (
                  <ResultCard
                    label="Color — Volume Cover Style"
                    src={`data:image/png;base64,${results.color}`}
                    onDownload={() => download("color")}
                  />
                )}
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                {isProcessing ? (
                  <div className="flex flex-col items-center gap-3">
                    <div className="w-8 h-8 border-2 border-zinc-600 border-t-zinc-200 rounded-full animate-spin" />
                    <p className="text-sm text-zinc-400">{stage || "Generating..."}</p>
                  </div>
                ) : (
                  <p className="text-zinc-600 text-sm">
                    {status === "idle" ? "Upload an image to begin" : "Results will appear here"}
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

function ResultCard({
  label,
  src,
  onDownload,
}: {
  label: string;
  src: string;
  onDownload: () => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      <div className="rounded-xl overflow-hidden bg-zinc-900 aspect-square">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={label} className="w-full h-full object-contain" />
      </div>
      <div className="flex items-center justify-between">
        <span className="text-xs text-zinc-400">{label}</span>
        <button
          onClick={onDownload}
          className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-zinc-800 text-zinc-300 hover:bg-zinc-700 transition-colors"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Download
        </button>
      </div>
    </div>
  );
}
