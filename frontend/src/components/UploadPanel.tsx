import { useRef, useState } from "react";

type UploadPanelProps = {
  disabled: boolean;
  previewUrl: string | null;
  selectedFileName: string | null;
  onFileSelected: (file: File) => void;
  onSubmit: () => void;
};

export function UploadPanel({
  disabled,
  previewUrl,
  selectedFileName,
  onFileSelected,
  onSubmit,
}: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (!file) {
      return;
    }
    if (!file.type.startsWith("image/")) {
      return;
    }
    onFileSelected(file);
  };

  return (
    <section className="rounded-[2rem] border border-white/10 bg-white/5 p-5 shadow-panel backdrop-blur md:p-7">
      <div
        className={`relative flex min-h-[320px] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-[1.5rem] border border-dashed p-6 text-center transition ${
          isDragging
            ? "border-aurora bg-aurora/10"
            : "border-white/20 bg-night-900/70 hover:border-white/40 hover:bg-night-800/80"
        }`}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          setIsDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          handleFiles(event.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          className="hidden"
          type="file"
          accept="image/*"
          onChange={(event) => handleFiles(event.target.files)}
        />

        {previewUrl ? (
          <img
            src={previewUrl}
            alt="Night sky preview"
            className="h-full max-h-[320px] w-full rounded-[1.2rem] object-cover"
          />
        ) : (
          <div className="space-y-4">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full border border-white/10 bg-white/10 text-2xl text-gold">
              *
            </div>
            <div>
              <p className="font-display text-2xl text-white">Drop a night-sky image here</p>
              <p className="mt-2 text-sm text-slate-300">
                PNG, JPG, or WEBP. Higher contrast sky photos produce better pattern matches.
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-white">Selected file</p>
          <p className="text-sm text-slate-300">{selectedFileName ?? "No image selected yet"}</p>
        </div>
        <button
          type="button"
          onClick={onSubmit}
          disabled={disabled}
          className="inline-flex items-center justify-center rounded-full bg-gradient-to-r from-gold to-coral px-5 py-3 font-semibold text-night-950 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Identify Constellations
        </button>
      </div>
    </section>
  );
}
