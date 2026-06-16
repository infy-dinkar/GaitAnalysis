"use client";
import { useCallback, useEffect, useRef, useState } from "react";

/** Secondary-camera hook for the gait record-live screen. Plain
 *  preview only — no MediaRecorder, no blob, no upload path.
 *  Caller owns the <video> element ref; we attach the stream to it
 *  on start() and detach on stop().
 *
 *  Separate from useCamera so that hook (10 other capture
 *  components depend on it) stays byte-identical. This one takes
 *  an explicit deviceId — no facingMode fallback because the
 *  secondary tile is only meaningful after the operator picks a
 *  device from the enumeration dropdown.
 */
export function useSecondaryCamera(
  videoRef: React.RefObject<HTMLVideoElement | null>,
) {
  const streamRef = useRef<MediaStream | null>(null);
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setActive(false);
  }, [videoRef]);

  const start = useCallback(
    async (deviceId: string) => {
      setError(null);
      if (!deviceId) {
        setError("No reference camera selected.");
        setActive(false);
        return;
      }
      // Stop any previous stream first — calling start() with a
      // different deviceId should swap, not stack.
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      try {
        // Minimal constraints (deviceId only — no width/height
        // hints) so virtual cameras like DroidCam / OBS Virtual /
        // SnapCamera negotiate at their native output resolution
        // instead of returning a stream whose frame pump never
        // starts when the hint can't be honoured.
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: deviceId } },
          audio: false,
        });
        streamRef.current = stream;
        if (videoRef.current) {
          const v = videoRef.current;
          v.srcObject = stream;
          // Some browsers/virtual cams need an explicit play()
          // AFTER metadata is parsed, not just the initial call —
          // the initial play() can race with stream-format
          // negotiation and leave the element wedged on a green
          // (or grey) placeholder.
          v.onloadedmetadata = () => {
            v.play().catch(() => {});
          };
          await v.play().catch(() => {});
        }
        setActive(true);
      } catch (e) {
        const msg =
          e instanceof Error ? e.message : "Reference camera access denied";
        setError(msg);
        setActive(false);
      }
    },
    [videoRef],
  );

  useEffect(() => stop, [stop]);

  return { streamRef, active, error, start, stop };
}
