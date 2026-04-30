// Movement-specific recording instructions for the biomech live + upload flows.
// Mirrors MOVEMENT_INSTRUCTIONS in biomech_flow.py — same source: AAOS / APTA
// standard ROM positioning guidance.

type Key = `${"shoulder" | "neck"}.${string}`;

export const MOVEMENT_INSTRUCTIONS: Record<Key, string[]> = {
  // ── Shoulder ──────────────────────────────────────────────────
  "shoulder.flexion": [
    "Stand 6 feet from the camera, facing sideways (true side view).",
    "Keep the arm being assessed straight, hanging at your side.",
    "Slowly raise the arm forward and upward as high as you can.",
    "Hold at the peak position for 2 seconds.",
    "Slowly lower the arm back to the starting position.",
  ],
  "shoulder.extension": [
    "Stand 6 feet from the camera, facing sideways (true side view).",
    "Keep the arm straight at your side as the starting position.",
    "Slowly reach the arm backward, behind your body, as far as comfortable.",
    "Hold at the peak position for 2 seconds.",
    "Return slowly to the starting position.",
  ],
  "shoulder.abduction": [
    "Stand 6 feet from the camera, facing the camera (front view).",
    "Keep the arm straight at your side, palm facing your body.",
    "Slowly raise the arm out to the side, palm down, as high as you can.",
    "Hold at the peak position for 2 seconds.",
    "Slowly lower back to the starting position.",
  ],
  "shoulder.adduction": [
    "Stand 6 feet from the camera, facing the camera (front view).",
    "Start with the arm raised slightly out to the side.",
    "Slowly bring the arm across the front of the body toward the opposite shoulder.",
    "Hold at the peak position for 2 seconds.",
    "Return slowly to the starting position.",
  ],
  "shoulder.external_rotation": [
    "Stand 6 feet from the camera, facing the camera (front view).",
    "Bend the elbow to 90°, with the upper arm at your side, forearm pointing forward.",
    "Without moving the elbow, slowly rotate the forearm outward (away from your body).",
    "Hold at the peak position for 2 seconds.",
    "Return slowly to the starting position.",
  ],
  "shoulder.internal_rotation": [
    "Stand 6 feet from the camera, facing the camera (front view).",
    "Bend the elbow to 90°, with the upper arm at your side, forearm pointing forward.",
    "Without moving the elbow, slowly rotate the forearm inward (across your stomach).",
    "Hold at the peak position for 2 seconds.",
    "Return slowly to the starting position.",
  ],

  // ── Neck (4 movements — engine has no per-side parameter) ─────
  "neck.flexion": [
    "Sit upright facing sideways to the camera (true side view).",
    "Keep your shoulders relaxed and still.",
    "Slowly bring your chin toward your chest as far as comfortable.",
    "Hold for 2 seconds at the peak.",
    "Return slowly to the upright neutral position.",
  ],
  "neck.extension": [
    "Sit upright facing sideways to the camera (true side view).",
    "Keep your shoulders relaxed and still.",
    "Slowly tilt your head backward to look up at the ceiling.",
    "Hold for 2 seconds at the peak.",
    "Return slowly to the upright neutral position.",
  ],
  "neck.lateral_flexion": [
    "Sit upright facing the camera (front view).",
    "Keep your shoulders level and relaxed — do not raise the shoulder toward the ear.",
    "Tilt your head so the ear moves toward the shoulder (do not turn the head — keep nose facing forward).",
    "Hold for 2 seconds, then return to upright.",
    "Repeat to the other side if assessing both.",
  ],
  "neck.rotation": [
    "Sit upright facing the camera (front view).",
    "Keep your shoulders level and still.",
    "Slowly turn your head to one side as far as comfortable, keeping the chin level.",
    "Hold for 2 seconds, then return to centre.",
    "Repeat to the other side if assessing both.",
  ],
};

export function getInstructions(
  bodyPart: "shoulder" | "neck",
  movementId: string,
): string[] {
  return MOVEMENT_INSTRUCTIONS[`${bodyPart}.${movementId}` as Key] ?? [];
}

export function isRotationMovement(
  bodyPart: "shoulder" | "neck",
  movementId: string,
): boolean {
  if (bodyPart === "shoulder") {
    return movementId === "external_rotation" || movementId === "internal_rotation";
  }
  return movementId === "rotation";
}
