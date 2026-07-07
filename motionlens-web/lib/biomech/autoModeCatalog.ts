// Auto-mode catalog — flattened list of every (joint, movement, side)
// triple the live camera can currently run. Consumed by:
//   • /biomech/auto — configurator UI (multi-select joints + moves)
//   • /biomech/auto/run — sequence runner that mounts LiveAssessment
//     per step with a countdown timer.
//
// Movement IDs are the SAME strings the live pages already accept via
// the ?movement=... query param (see /biomech/{joint}/live/page.tsx),
// so no engine wiring changes are needed. This file is pure metadata.

import type { NeckMovement } from "./neck-live";

export type Joint = "neck" | "shoulder" | "knee" | "hip" | "ankle";
export type Side = "left" | "right";

export interface JointMeta {
  id: Joint;
  label: string;
  gradient: string;
  /** true = movements are unilateral and need a side (shoulder/knee/hip/ankle).
   *  false = bilateral by nature (neck). */
  hasSide: boolean;
}

export const JOINT_META: JointMeta[] = [
  {
    id: "neck",
    label: "Neck",
    hasSide: false,
    gradient:
      "linear-gradient(135deg, rgba(255,183,77,0.18) 0%, rgba(234,88,12,0.10) 50%, rgba(28,28,33,0.0) 100%)",
  },
  {
    id: "shoulder",
    label: "Shoulder",
    hasSide: true,
    gradient:
      "linear-gradient(135deg, rgba(234,88,12,0.18) 0%, rgba(79,195,247,0.10) 50%, rgba(28,28,33,0.0) 100%)",
  },
  {
    id: "hip",
    label: "Hip",
    hasSide: true,
    gradient:
      "linear-gradient(135deg, rgba(168,85,247,0.18) 0%, rgba(234,88,12,0.10) 50%, rgba(28,28,33,0.0) 100%)",
  },
  {
    id: "knee",
    label: "Knee",
    hasSide: true,
    gradient:
      "linear-gradient(135deg, rgba(79,195,247,0.20) 0%, rgba(56,189,248,0.10) 50%, rgba(28,28,33,0.0) 100%)",
  },
  {
    id: "ankle",
    label: "Ankle",
    hasSide: true,
    gradient:
      "linear-gradient(135deg, rgba(34,197,94,0.18) 0%, rgba(79,195,247,0.10) 50%, rgba(28,28,33,0.0) 100%)",
  },
];

export interface MovementDef {
  /** Same id string that /biomech/{joint}/live accepts via ?movement=… */
  id: string;
  /** Human label for the UI. */
  label: string;
  /** Short description of the movement's action. */
  description: string;
  /** Normal ROM target band (deg) — pulled from the live modules'
   *  own MOVEMENTS tables so the runner passes the same value the
   *  standalone /biomech/{joint}/live page would. */
  target: [number, number];
}

/** Movements each joint offers in the live camera flow. Kept flat to
 *  simplify the picker UI — one row per movement. */
export const MOVEMENTS_BY_JOINT: Record<Joint, MovementDef[]> = {
  neck: [
    {
      id: "flexion_extension",
      label: "Flexion + Extension",
      description: "Tilt head forward and backward",
      target: [45, 80],
    },
    {
      id: "lateral_flexion",
      label: "Lateral Flexion",
      description: "Ear to shoulder, both sides",
      target: [20, 45],
    },
    {
      id: "rotation",
      label: "Rotation",
      description: "Turn head left and right",
      target: [70, 90],
    },
  ],
  shoulder: [
    {
      id: "flexion_extension",
      label: "Flexion + Extension",
      description: "Arm forward + arm backward",
      target: [150, 180],
    },
    {
      id: "abduction_adduction",
      label: "Abduction + Adduction",
      description: "Arm sideways + across chest",
      target: [150, 180],
    },
    {
      id: "rotation",
      label: "External + Internal Rotation",
      description: "Elbow at 90° — rotate forearm",
      target: [70, 90],
    },
  ],
  hip: [
    {
      id: "flexion",
      label: "Flexion",
      description: "Lift knee toward chest",
      target: [110, 130],
    },
    {
      id: "extension",
      label: "Extension",
      description: "Extend leg backward",
      target: [10, 30],
    },
    {
      id: "rotation",
      label: "Rotation (Internal + External)",
      description: "Seated — rotate lower leg in/out",
      target: [30, 45],
    },
  ],
  knee: [
    {
      id: "flexion_extension",
      label: "Flexion + Extension",
      description: "Bend and straighten the knee",
      target: [125, 145],
    },
  ],
  ankle: [
    {
      id: "flexion",
      label: "Dorsiflexion",
      description: "Foot up, toes toward shin",
      target: [15, 25],
    },
    {
      id: "extension",
      label: "Plantarflexion",
      description: "Foot down, like pressing pedal",
      target: [40, 55],
    },
  ],
};

export interface AutoStep {
  joint: Joint;
  movementId: string;
  movementLabel: string;
  description: string;
  target: [number, number];
  /** null for bilateral joints (neck). */
  side: Side | null;
}

/**
 * Build the queue from the user's picks.
 *
 * @param picks     — Map of joint → set of selected movement ids
 * @param sides     — Set of selected sides. Ignored for bilateral joints.
 */
export function buildAutoQueue(
  picks: Map<Joint, Set<string>>,
  sides: Set<Side>,
): AutoStep[] {
  const queue: AutoStep[] = [];
  // Iterate in the JOINT_META order so the queue always runs
  // top-to-bottom of the standard joint order rather than in map
  // insertion order.
  for (const meta of JOINT_META) {
    const chosenMoves = picks.get(meta.id);
    if (!chosenMoves || chosenMoves.size === 0) continue;
    const availableMoves = MOVEMENTS_BY_JOINT[meta.id];
    for (const move of availableMoves) {
      if (!chosenMoves.has(move.id)) continue;
      if (!meta.hasSide) {
        queue.push({
          joint: meta.id,
          movementId: move.id,
          movementLabel: move.label,
          description: move.description,
          target: move.target,
          side: null,
        });
        continue;
      }
      // Unilateral: expand to selected sides. If somehow no side is
      // set, default to a single "left" run so the queue is never
      // silently empty for a picked movement.
      const sideList: Side[] =
        sides.size > 0
          ? (Array.from(sides) as Side[]).sort()
          : ["left"];
      for (const s of sideList) {
        queue.push({
          joint: meta.id,
          movementId: move.id,
          movementLabel: move.label,
          description: move.description,
          target: move.target,
          side: s,
        });
      }
    }
  }
  return queue;
}

/**
 * Encode a queue into a compact URL-safe string so the runner page
 * can pick it up via ?q=... . Format:
 *   joint|move|side  triples joined by ";"
 *   side="-" when null.
 */
export function encodeQueue(queue: AutoStep[]): string {
  return queue
    .map((s) => `${s.joint}|${s.movementId}|${s.side ?? "-"}`)
    .join(";");
}

/** Reverse of encodeQueue. Silently drops malformed entries. */
export function decodeQueue(encoded: string): AutoStep[] {
  if (!encoded) return [];
  const out: AutoStep[] = [];
  for (const raw of encoded.split(";")) {
    const parts = raw.split("|");
    if (parts.length !== 3) continue;
    const [joint, movementId, sideRaw] = parts;
    const meta = JOINT_META.find((m) => m.id === joint);
    if (!meta) continue;
    const move = MOVEMENTS_BY_JOINT[meta.id].find((m) => m.id === movementId);
    if (!move) continue;
    const side: Side | null =
      sideRaw === "left" || sideRaw === "right" ? sideRaw : null;
    out.push({
      joint: meta.id,
      movementId: move.id,
      movementLabel: move.label,
      description: move.description,
      target: move.target,
      side,
    });
  }
  return out;
}

/** Human title for a single step — used in the runner's header. */
export function stepTitle(s: AutoStep): string {
  const meta = JOINT_META.find((m) => m.id === s.joint);
  const jointLbl = meta?.label ?? s.joint;
  const sideLbl = s.side
    ? ` · ${s.side === "left" ? "Left" : "Right"}`
    : "";
  return `${jointLbl} · ${s.movementLabel}${sideLbl}`;
}

// Suppress unused import warning — kept as a type-check the neck
// module's shape stays compatible with our metadata.
export type _neck = NeckMovement;
