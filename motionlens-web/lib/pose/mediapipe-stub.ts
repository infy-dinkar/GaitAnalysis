// Stub for @mediapipe/pose. The pose-detection package statically imports
// `Pose` from this module to support its mediapipe runtime, but the upload
// flow uses the pure tfjs runtime so the class is never instantiated. The
// stub only exists to satisfy ESM resolution at build time — the real
// @mediapipe/pose `pose.js` is UMD-only with no ESM exports, which
// Turbopack rejects.
//
// The live stack (lib/pose/detector-live.ts + hooks/usePoseDetectionLive.ts)
// uses `import type` for compile-time types only (erased at bundle time)
// and loads the actual UMD bundle via a runtime <script> tag, so it
// bypasses this alias entirely.
export class Pose {}
