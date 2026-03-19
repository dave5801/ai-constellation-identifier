import { useEffect, useState } from "react";
import { ResultsPanel } from "./components/ResultsPanel";
import { UploadPanel } from "./components/UploadPanel";
import { identifySkyImage } from "./lib/api";
import type { IdentifyResponse } from "./lib/types";

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<IdentifyResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  async function handleSubmit() {
    if (!selectedFile) {
      setError("Choose a night-sky image before starting analysis.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await identifySkyImage(selectedFile);
      setResult(response);
    } catch (submitError) {
      const message =
        submitError instanceof Error ? submitError.message : "Unexpected analysis error.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-night-950 bg-grid bg-[size:100%_100%,32px_32px,32px_32px] text-white">
      <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        <header className="max-w-3xl">
          <p className="text-sm uppercase tracking-[0.38em] text-aurora">Computer Vision Astronomy</p>
          <h1 className="mt-4 font-display text-4xl leading-tight text-white sm:text-6xl">
            AI Constellation & Object Identifier
          </h1>
          <p className="mt-5 text-base leading-7 text-slate-300 sm:text-lg">
            Upload a night-sky photo to detect bright stars, identify supported constellations,
            estimate confidence, and review an annotated image overlay built by the FastAPI backend.
          </p>
        </header>

        <main className="mt-10 grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
          <UploadPanel
            disabled={isLoading}
            previewUrl={previewUrl}
            selectedFileName={selectedFile?.name ?? null}
            onFileSelected={(file) => {
              setSelectedFile(file);
              setResult(null);
              setError(null);
            }}
            onSubmit={handleSubmit}
          />
          <ResultsPanel result={result} isLoading={isLoading} error={error} />
        </main>
      </div>
    </div>
  );
}

export default App;
