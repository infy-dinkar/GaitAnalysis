import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
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
