// Stub for @mediapipe/pose. The pose-detection package statically imports
// `Pose` from this module to support its mediapipe runtime, but we run on
// the pure tfjs runtime so the class is never instantiated. The stub only
// exists to satisfy ESM resolution at build time.
export class Pose {}
