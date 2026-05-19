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
  /** Merged-test fallback fields. Populated only for shoulder
   *  "rotation" and "abduction_adduction" — saved-report viewer uses
   *  these when the saved metrics blob doesn't carry primary_label /
   *  secondary_label inline (older records saved before merged-test
   *  support landed). */
  merged?: boolean;
  primaryLabel?: string;
  secondaryLabel?: string;
  secondaryTarget?: [number, number];
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
  if (!m) return null;
  // SHOULDER_MOVEMENTS entries carry the merged-test fields directly;
  // the spread keeps everything other than the legacy two-field shape
  // safely passed through. The other body-part metadata tables only
  // expose label + target so this just doesn't pick up the extra
  // fields for them.
  const result: MovementMeta = { label: m.label, target: m.target };
  const shoulderEntry = m as typeof m & {
    merged?: boolean;
    primaryLabel?: string;
    secondaryLabel?: string;
    secondaryTarget?: [number, number];
  };
  if (shoulderEntry.merged) result.merged = true;
  if (shoulderEntry.primaryLabel) result.primaryLabel = shoulderEntry.primaryLabel;
  if (shoulderEntry.secondaryLabel) result.secondaryLabel = shoulderEntry.secondaryLabel;
  if (shoulderEntry.secondaryTarget) result.secondaryTarget = shoulderEntry.secondaryTarget;
  return result;
}
