import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // Allow LAN access during dev (e.g. testing on phone via local IP).
  // Next.js 16 blocks cross-origin HMR by default; whitelist the LAN
  // subnet so the dev server works when accessed by IP, not just
  // localhost.
  allowedDevOrigins: ["10.108.169.255", "192.168.1.0/24", "10.0.0.0/8"],

  // pose-detection statically imports `Pose` from @mediapipe/pose, which has
  // no ESM exports. We use the tfjs runtime so the mediapipe class is never
  // instantiated — alias it to a local stub.
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
