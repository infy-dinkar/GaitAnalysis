// Resolve a (bodyPart, movementId) pair back to its human-readable label.
// Used by the saved-report viewer where we only have the IDs from the DB.

import { SHOULDER_MOVEMENTS } from "@/lib/biomech/shoulder";
import { NECK_MOVEMENTS } from "@/lib/biomech/neck";
import { KNEE_MOVEMENTS } from "@/lib/biomech/knee";
import { HIP_MOVEMENTS } from "@/lib/biomech/hip";
import { ANKLE_MOVEMENTS } from "@/lib/biomech/ankle";

type BodyPart = "shoulder" | "neck" | "knee" | "hip" | "ankle";

interface MovementMeta {
  label: string;
  target: [number, number];
}

export function resolveMovement(
  bodyPart: BodyPart,
  movementId: string,
): MovementMeta | null {
  const list =
    bodyPart === "shoulder" ? SHOULDER_MOVEMENTS
    : bodyPart === "neck"   ? NECK_MOVEMENTS
    : bodyPart === "knee"   ? KNEE_MOVEMENTS
    : bodyPart === "hip"    ? HIP_MOVEMENTS
    : ANKLE_MOVEMENTS;
  const m = list.find((x) => x.id === movementId);
  return m ? { label: m.label, target: m.target } : null;
}
