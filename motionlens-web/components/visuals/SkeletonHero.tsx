"use client";
import { motion } from "framer-motion";

// Front-view humanoid skeleton with anatomical color-coding.
//   • Spine (head → neck → mid → pelvis):  lime accent
//   • Left side limbs:                     cyan blue
//   • Right side limbs:                    rose red
//   • Head silhouette:                     warm gold
//   • Pelvis bar:                          violet
// The colour split mirrors the chart palette used elsewhere in the app
// (left = blue, right = red), so the figure tells the same visual story
// as the gait time-series charts.

const COLORS = {
  head: "#FBBF24",
  spine: "#EA580C",
  pelvis: "#A78BFA",
  left: "#4FC3F7",
  right: "#FB7185",
} as const;

type ColorKey = keyof typeof COLORS;

interface Joint {
  x: number;
  y: number;
  color: ColorKey;
}

// ViewBox 400×500. Figure spans y ≈ 60–470 with comfortable padding.
const joints: Joint[] = [
  /*  0 */ { x: 200, y: 70,  color: "head" },     // head
  /*  1 */ { x: 200, y: 130, color: "spine" },    // neck
  /*  2 */ { x: 155, y: 148, color: "left" },     // L shoulder
  /*  3 */ { x: 245, y: 148, color: "right" },    // R shoulder
  /*  4 */ { x: 122, y: 222, color: "left" },     // L elbow
  /*  5 */ { x: 278, y: 222, color: "right" },    // R elbow
  /*  6 */ { x: 102, y: 296, color: "left" },     // L wrist
  /*  7 */ { x: 298, y: 296, color: "right" },    // R wrist
  /*  8 */ { x: 200, y: 280, color: "spine" },    // mid-spine
  /*  9 */ { x: 175, y: 295, color: "pelvis" },   // L hip
  /* 10 */ { x: 225, y: 295, color: "pelvis" },   // R hip
  /* 11 */ { x: 168, y: 388, color: "left" },     // L knee
  /* 12 */ { x: 232, y: 388, color: "right" },    // R knee
  /* 13 */ { x: 162, y: 478, color: "left" },     // L ankle
  /* 14 */ { x: 238, y: 478, color: "right" },    // R ankle
];

interface Edge {
  a: number;
  b: number;
  color: ColorKey;
}

const edges: Edge[] = [
  // spine
  { a: 0, b: 1, color: "spine" },
  { a: 1, b: 8, color: "spine" },
  // shoulders
  { a: 2, b: 3, color: "spine" },
  { a: 1, b: 2, color: "left" },
  { a: 1, b: 3, color: "right" },
  // left arm
  { a: 2, b: 4, color: "left" },
  { a: 4, b: 6, color: "left" },
  // right arm
  { a: 3, b: 5, color: "right" },
  { a: 5, b: 7, color: "right" },
  // pelvis
  { a: 8, b: 9, color: "pelvis" },
  { a: 8, b: 10, color: "pelvis" },
  { a: 9, b: 10, color: "pelvis" },
  // left leg
  { a: 9, b: 11, color: "left" },
  { a: 11, b: 13, color: "left" },
  // right leg
  { a: 10, b: 12, color: "right" },
  { a: 12, b: 14, color: "right" },
];

export function SkeletonHero() {
  return (
    <div className="relative h-full w-full">
      {/* ambient glow */}
      <div className="absolute left-1/2 top-1/2 h-[58%] w-[58%] -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent/10 blur-3xl" />

      <svg
        viewBox="0 0 400 500"
        className="relative h-full w-full"
        preserveAspectRatio="xMidYMid meet"
        aria-hidden
      >
        <defs>
          {(Object.keys(COLORS) as ColorKey[]).map((k) => (
            <radialGradient key={k} id={`glow-${k}`} cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor={COLORS[k]} stopOpacity="0.55" />
              <stop offset="100%" stopColor={COLORS[k]} stopOpacity="0" />
            </radialGradient>
          ))}
          <radialGradient id="head-glow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor={COLORS.head} stopOpacity="0.35" />
            <stop offset="100%" stopColor={COLORS.head} stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* head silhouette: outer glow + outline */}
        <circle cx={200} cy={70} r={36} fill="url(#head-glow)" />
        <motion.circle
          cx={200}
          cy={70}
          r={22}
          fill="none"
          stroke={COLORS.head}
          strokeWidth={1.2}
          strokeOpacity={0.5}
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 0.55 }}
          transition={{ delay: 0.05, duration: 0.5, ease: "easeOut" }}
        />

        {/* edges */}
        {edges.map(({ a, b, color }, i) => {
          const p1 = joints[a];
          const p2 = joints[b];
          return (
            <motion.line
              key={i}
              x1={p1.x}
              y1={p1.y}
              x2={p2.x}
              y2={p2.y}
              stroke={COLORS[color]}
              strokeWidth={1.8}
              strokeOpacity={0.65}
              strokeLinecap="round"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.75 }}
              transition={{
                delay: 0.2 + i * 0.06,
                duration: 0.6,
                ease: "easeOut",
              }}
            />
          );
        })}

        {/* joints — outer glow disc, pulsing dot, dark inner */}
        {joints.map((j, i) => (
          <g key={i}>
            <circle cx={j.x} cy={j.y} r={18} fill={`url(#glow-${j.color})`} />
            <motion.circle
              cx={j.x}
              cy={j.y}
              fill={COLORS[j.color]}
              initial={{ r: 0, opacity: 0 }}
              animate={{ r: [4.8, 6, 4.8], opacity: [1, 0.7, 1] }}
              transition={{
                r: { duration: 2.4, repeat: Infinity, delay: 0.4 + i * 0.08 },
                opacity: {
                  duration: 2.4,
                  repeat: Infinity,
                  delay: 0.4 + i * 0.08,
                },
              }}
            />
            <circle cx={j.x} cy={j.y} r={1.5} fill="#0A0A0B" />
          </g>
        ))}

        {/* subtle motion trail along the right leg — suggests gait */}
        <motion.circle
          r={3.2}
          fill={COLORS.right}
          initial={{ opacity: 0 }}
          animate={{
            cx: [joints[10].x, joints[12].x, joints[14].x, joints[12].x, joints[10].x],
            cy: [joints[10].y, joints[12].y, joints[14].y, joints[12].y, joints[10].y],
            opacity: [0, 0.85, 0.85, 0.85, 0],
          }}
          transition={{ duration: 4.5, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.circle
          r={3.2}
          fill={COLORS.left}
          initial={{ opacity: 0 }}
          animate={{
            cx: [joints[9].x, joints[11].x, joints[13].x, joints[11].x, joints[9].x],
            cy: [joints[9].y, joints[11].y, joints[13].y, joints[11].y, joints[9].y],
            opacity: [0, 0.85, 0.85, 0.85, 0],
          }}
          transition={{
            duration: 4.5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 2.25,
          }}
        />
      </svg>
    </div>
  );
}
