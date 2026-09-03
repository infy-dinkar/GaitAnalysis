// Resolve a (bodyPart, movementId) pair back to its human-readable label.
// Used by the saved-report viewer where we only have the IDs from the DB.

import { SHOULDER_MOVEMENTS } from "@/lib/biomech/shoulder";
import { NECK_MOVEMENTS } from "@/lib/biomech/neck";
import { KNEE_MOVEMENTS } from "@/lib/biomech/knee";
import { HIP_MOVEMENTS } from "@/lib/biomech/hip";
import { ANKLE_MOVEMENTS } from "@/lib/biomech/ankle";

type BodyPart = "shoulder" | "neck" | "knee" | "hip" | "ankle";

/** Deprecated single-direction movement IDs → the merged test that
 *  replaced them.
 *
 *  These IDs stay resolvable (see resolveMovement below) so SAVED
 *  REPORTS that recorded them keep rendering their label + reference
 *  range. They must never start a NEW analysis though: the upload
 *  pipeline has no backend branch for them, and they used to fall
 *  through to an in-browser MoveNet analyser running a different model
 *  with a different keypoint set. Upload pages look the requested
 *  movement up here and silently redirect to the merged equivalent.
 *
 *  Shoulder and ankle legacy IDs are deliberately absent: the backend
 *  accepts those verbatim, so they still analyse correctly as-is. */
export const LEGACY_MOVEMENT_MAP: Record<string, Record<string, string>> = {
  knee: {
    flexion: "flexion_extension",
    extension: "flexion_extension",
  },
  neck: {
    flexion: "flexion_extension",
    extension: "flexion_extension",
  },
  hip: {
    internal_rotation: "rotation",
    external_rotation: "rotation",
  },
};

/** The merged replacement for a deprecated ID, or null when the ID is
 *  still analysable as requested. */
export function resolveLegacyMovement(
  bodyPart: string,
  movementId: string,
): string | null {
  return LEGACY_MOVEMENT_MAP[bodyPart]?.[movementId] ?? null;
}

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
  /** Optional reference illustration. Lives on the joint movement
   *  entry (e.g. SHOULDER_MOVEMENTS); passed through here so callers
   *  like LiveAssessment can resolve it without importing each
   *  joint's array directly. */
  imageUrl?: string;
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
    imageUrl?: string;
  };
  if (shoulderEntry.merged) result.merged = true;
  if (shoulderEntry.primaryLabel) result.primaryLabel = shoulderEntry.primaryLabel;
  if (shoulderEntry.secondaryLabel) result.secondaryLabel = shoulderEntry.secondaryLabel;
  if (shoulderEntry.secondaryTarget) result.secondaryTarget = shoulderEntry.secondaryTarget;
  if (shoulderEntry.imageUrl) result.imageUrl = shoulderEntry.imageUrl;
  return result;
}
