import type { VideoInfoDTO } from "@/lib/api";
import { fmt } from "@/lib/utils";

interface Props {
  videoInfo: VideoInfoDTO;
  heightCm: number;
}

export function CalibrationHeader({ videoInfo, heightCm }: Props) {
  const cal =
    videoInfo.calibration_mm_per_px !== null
      ? `${fmt(videoInfo.calibration_mm_per_px, 3)} mm/px`
      : "uncalibrated";

  return (
    <div className="space-y-1 text-xs text-subtle">
      <p>
        Processed{" "}
        <span className="tabular text-foreground">{videoInfo.total_frames}</span> frames at{" "}
        <span className="tabular text-foreground">{fmt(videoInfo.fps, 0)}</span> FPS —{" "}
        <span className="tabular text-foreground">{fmt(videoInfo.duration_sec, 1)}s</span> total
      </p>
      <p>
        Calibration: <span className="tabular text-foreground">{cal}</span> ·{" "}
        <span className="tabular text-foreground">{videoInfo.valid_passes}</span> valid passes ·{" "}
        <span className="tabular text-foreground">
          {videoInfo.frames_used}/{videoInfo.total_frames}
        </span>{" "}
        frames used · height ={" "}
        <span className="tabular text-foreground">{fmt(heightCm, 0)} cm</span>
      </p>
      <p>
        Ankle baseline correction: L ={" "}
        <span className="tabular text-foreground">
          {fmt(videoInfo.ankle_baseline_left, 1)}°
        </span>
        , R ={" "}
        <span className="tabular text-foreground">
          {fmt(videoInfo.ankle_baseline_right, 1)}°
        </span>{" "}
        ({videoInfo.ankle_baseline_method}, n ={" "}
        <span className="tabular text-foreground">
          {videoInfo.ankle_baseline_n_frames}
        </span>{" "}
        frames)
      </p>
    </div>
  );
}
