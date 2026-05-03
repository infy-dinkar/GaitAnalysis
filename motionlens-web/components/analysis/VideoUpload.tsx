"use client";
import { ChangeEvent, DragEvent, useRef, useState } from "react";
import { Upload, FileVideo, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface VideoUploadProps {
  onSelect: (file: File) => void;
  accept?: string;
  className?: string;
}

export function VideoUpload({ onSelect, accept = "video/*", className }: VideoUploadProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [drag, setDrag] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  function handleFile(f: File | null | undefined) {
    if (!f) return;
    setFile(f);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
    onSelect(f);
  }

  function onDrop(e: DragEvent<HTMLLabelElement>) {
    e.preventDefault();
    setDrag(false);
    handleFile(e.dataTransfer.files?.[0]);
  }

  function onChange(e: ChangeEvent<HTMLInputElement>) {
    handleFile(e.target.files?.[0]);
  }

  function clear() {
    setFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  if (file && previewUrl) {
    return (
      <div className={cn("rounded-card border border-border bg-surface p-4", className)}>
        <div className="flex items-center justify-between gap-3 pb-3">
          <div className="flex min-w-0 items-center gap-2">
            <FileVideo className="h-4 w-4 shrink-0 text-accent" />
            <span className="truncate text-sm text-foreground">{file.name}</span>
            <span className="shrink-0 text-xs text-subtle">
              {(file.size / 1024 / 1024).toFixed(1)} MB
            </span>
          </div>
          <button
            type="button"
            onClick={clear}
            className="text-muted transition hover:text-foreground"
            aria-label="Remove video"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <video
          src={previewUrl}
          controls
          playsInline
          className="aspect-video w-full rounded-lg bg-background"
        />
      </div>
    );
  }

  return (
    <label
      htmlFor="video-upload-input"
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
      className={cn(
        "group relative flex cursor-pointer flex-col items-center justify-center gap-3 rounded-card border-2 border-dashed bg-surface px-6 py-16 text-center transition-all duration-200",
        drag ? "border-accent bg-accent/5 shadow-glow-sm" : "border-border hover:border-accent",
        className,
      )}
    >
      <span className="flex h-14 w-14 items-center justify-center rounded-full bg-accent/10 text-accent transition group-hover:scale-110">
        <Upload className="h-6 w-6" />
      </span>
      <div>
        <p className="text-sm font-medium text-foreground">
          {drag ? "Drop the video here" : "Click or drag a video to upload"}
        </p>
        <p className="mt-1 text-xs text-muted">MP4, MOV, WebM — up to ~200 MB</p>
      </div>
      <input
        id="video-upload-input"
        ref={inputRef}
        type="file"
        accept={accept}
        className="sr-only"
        onChange={onChange}
      />
    </label>
  );
}
