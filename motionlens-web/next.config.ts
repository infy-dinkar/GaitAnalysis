import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // Allow LAN access during dev (e.g. testing on phone via local IP).
  // Next.js 16 blocks cross-origin HMR by default; whitelist the LAN
  // subnet so the dev server works when accessed by IP, not just
  // localhost.
  allowedDevOrigins: ["10.108.169.255", "192.168.1.0/24", "10.0.0.0/8"],

  // @tensorflow-models/pose-detection statically imports `Pose` from
  // @mediapipe/pose to support its mediapipe runtime. The real
  // @mediapipe/pose ships only a UMD bundle (`pose.js` registers
  // `window.Pose` — no ESM exports), which Turbopack rejects with
  // "Export Pose doesn't exist in target module". The upload path
  // uses pose-detection's tfjs runtime, so the mediapipe class is
  // never instantiated — alias it to a local empty stub that DOES
  // have a named export, and the static-import resolution is happy.
  //
  // The live stack (detector-live.ts) deliberately uses
  // `import type { Pose }` (TS-only, erased at bundle time) and
  // loads the real UMD bundle via a runtime <script> tag from CDN,
  // so it never goes through this alias.
  turbopack: {
    resolveAlias: {
      "@mediapipe/pose": "./lib/pose/mediapipe-stub.ts",
    },
  },
  webpack: (config) => {
    config.resolve.alias["@mediapipe/pose"] = path.resolve(
      __dirname,
      "lib/pose/mediapipe-stub.ts",
    );
    return config;
  },
};

export default nextConfig;
