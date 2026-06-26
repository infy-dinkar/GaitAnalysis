// Reference image map for rehab exercises. Single source of truth
// mapping slug → public URL. Mirrors the biomech pattern where each
// Movement entry carries an optional imageUrl that resolves to a
// file under public/images/biomech/<joint>/<file>.png.
//
// Rehab uses a flat folder under public/images/rehab/ keyed by the
// exercise slug (the same string used as the route folder name in
// app/rehab/<slug>/page.tsx and in catalogue + dashboard hrefs).
//
// Consumers (each rehab page + catalogue + dashboard) look up by
// slug; a missing entry returns undefined and the conditional
// render bails (no broken <img> tag). Matches biomech's
// `imageUrl?: string` opt-in semantics.

export const REHAB_EXERCISE_IMAGES: Record<string, string> = {
  "squat": "/images/rehab/squat.png",
  "wall-sit": "/images/rehab/wall-sit.png",
  "pelvic-hold": "/images/rehab/pelvic-hold.png",
  "shoulder-raise": "/images/rehab/shoulder-raise.png",
  "pendulum": "/images/rehab/pendulum.png",
  "weight-shift": "/images/rehab/weight-shift.png",
  "bridge": "/images/rehab/bridge.png",
  "step-up": "/images/rehab/step-up.png",
  "lateral-step": "/images/rehab/lateral-step.png",
  "knee-extension": "/images/rehab/knee-extension.png",
  "single-leg-squat": "/images/rehab/single-leg-squat.png",
  "hip-abduction": "/images/rehab/hip-abduction.png",
  "wall-slide": "/images/rehab/wall-slide.png",
  "wall-clock": "/images/rehab/wall-clock.png",
  "external-rotation": "/images/rehab/external-rotation.png",
  "scapular-set": "/images/rehab/scapular-set.png",
};
