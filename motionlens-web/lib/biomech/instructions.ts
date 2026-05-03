// Movement-specific recording instructions for the biomech live + upload flows.
// Written in plain, friendly language so a non-clinical person can follow along
// without prior physio knowledge.

type Key = `${"shoulder" | "neck" | "knee" | "hip" | "ankle"}.${string}`;

export const MOVEMENT_INSTRUCTIONS: Record<Key, string[]> = {
  // ── Shoulder ──────────────────────────────────────────────────
  "shoulder.flexion": [
    "Stand about 6 feet (2 metres) from the camera so your full upper body is visible. Turn so the camera sees your side — your shoulder should point straight at the lens.",
    "Let the arm being measured hang straight down by your side. Keep the elbow straight and the palm facing your body.",
    "Slowly raise that arm forward and up, like you're reaching for the ceiling. Move smoothly — no jerks. Stop when it feels tight or you can't go higher.",
    "Hold at the highest point for about 2 seconds, keeping the elbow straight.",
    "Lower the arm back down to your side at the same slow speed. Keep your shoulders relaxed throughout — don't shrug.",
  ],
  "shoulder.extension": [
    "Stand about 6 feet (2 metres) from the camera, turned so the camera sees your side.",
    "Stand tall with the arm hanging straight by your side, palm facing your body. This is the starting position.",
    "Keeping the elbow straight, slowly move the arm backward behind your body — like you're reaching for something behind you. Don't lean forward to cheat the movement.",
    "Hold at the furthest comfortable point for 2 seconds.",
    "Slowly bring the arm back to your side. Stay tall and avoid arching your lower back.",
  ],
  "shoulder.abduction": [
    "Stand about 6 feet (2 metres) from the camera, facing the camera straight on so it can see both arms and your full torso.",
    "Let the arm hang at your side with the palm facing your body and the elbow straight.",
    "Slowly lift the arm out to the side — like making a snow-angel — keeping the palm facing down. Continue lifting until the arm is overhead, or until it feels tight.",
    "Hold the highest point for 2 seconds. Don't tilt your body to the opposite side to gain extra range.",
    "Lower the arm slowly back to your side along the same path.",
  ],
  "shoulder.adduction": [
    "Stand about 6 feet (2 metres) from the camera, facing the camera straight on.",
    "Start with the arm raised slightly out to the side (about 30°), elbow straight, palm facing down.",
    "Slowly bring the arm across the front of your chest toward the opposite shoulder, keeping the elbow straight.",
    "Hold at the furthest reach for 2 seconds. Keep your trunk facing forward — don't twist.",
    "Return the arm slowly to the starting position out to the side.",
  ],
  "shoulder.external_rotation": [
    "Stand about 6 feet (2 metres) from the camera, facing the camera straight on.",
    "Bend the elbow to 90° (right angle) and tuck the upper arm against your side. Forearm should point straight forward, like a forearm-shake position.",
    "Keeping the elbow tucked in and bent at 90°, slowly rotate the forearm outward — away from your body — like opening a door.",
    "Hold the furthest comfortable point for 2 seconds. Keep the elbow stuck to your side throughout — don't let it drift away.",
    "Slowly bring the forearm back to the front-facing starting position.",
  ],
  "shoulder.internal_rotation": [
    "Stand about 6 feet (2 metres) from the camera, facing the camera straight on.",
    "Bend the elbow to 90° (right angle) and tuck the upper arm against your side. Forearm should point straight forward.",
    "Keeping the elbow tucked in and bent at 90°, slowly rotate the forearm inward — across the front of your stomach.",
    "Hold the furthest comfortable point for 2 seconds. Don't let the elbow lift away from your side.",
    "Slowly bring the forearm back to the front-facing starting position.",
  ],

  // ── Neck (4 movements — engine has no per-side parameter) ─────
  "neck.flexion": [
    "Sit upright on a chair so your full head and shoulders are in frame. Turn so the camera sees the side of your head.",
    "Keep your shoulders relaxed and down — don't shrug. Look straight ahead with your chin level.",
    "Slowly lower your chin down toward your chest — like you're nodding 'yes' very slowly. Move only the head, not the upper back.",
    "Hold the lowest comfortable position for 2 seconds.",
    "Slowly bring the head back up to looking straight ahead.",
  ],
  "neck.extension": [
    "Sit upright on a chair, turned so the camera sees the side of your head.",
    "Keep your shoulders relaxed and your back tall. Look straight ahead.",
    "Slowly tilt your head back to look up at the ceiling. Move smoothly — stop if it feels uncomfortable or causes dizziness.",
    "Hold the furthest comfortable position for 2 seconds.",
    "Slowly bring the head back to facing forward.",
  ],
  "neck.lateral_flexion": [
    "Sit upright on a chair, facing the camera straight on so it can see both ears and both shoulders.",
    "Keep your shoulders level and relaxed — important: do NOT lift your shoulder up toward your ear. The ear comes to the shoulder, not the other way around.",
    "Slowly tip your head sideways so the ear moves toward the same-side shoulder. Keep the nose pointing forward — this is a tilt, not a turn.",
    "Hold for 2 seconds at the lowest comfortable point, then bring the head back to upright.",
    "If you want to measure both sides, repeat the movement to the opposite side.",
  ],
  "neck.rotation": [
    "Sit upright on a chair, facing the camera straight on so it can see both ears.",
    "Keep your shoulders level and still. Look straight ahead with your chin parallel to the floor.",
    "Slowly turn your head to look over one shoulder — keep the chin level (don't tilt or drop it). Stop when it feels tight.",
    "Hold for 2 seconds, then turn smoothly back to centre.",
    "If you want to measure both sides, repeat the movement to the opposite side.",
  ],

  // ── Knee ─────────────────────────────────────────────────────
  "knee.flexion": [
    "Stand or sit about 6 feet (2 metres) from the camera, turned so the camera sees the side of the leg you're testing.",
    "Stand tall with the leg straight as the starting position. The camera should see your hip, knee, and ankle clearly.",
    "Slowly lift the heel toward the buttock by bending the knee, keeping the thigh roughly vertical. Move smoothly.",
    "Hold the most-bent position comfortably for 2 seconds.",
    "Slowly lower the leg back to fully straight. Don't push past pain — measure pain-free range only.",
  ],
  "knee.extension": [
    "Sit on a chair so the camera sees the side of the leg you're testing. Hip, knee and ankle should all be visible.",
    "Start with the knee bent at about 90° (foot flat on the floor or hanging).",
    "Slowly straighten the knee by lifting the foot forward, until the leg is fully extended (or as far as you can go).",
    "Hold the straightest position for 2 seconds.",
    "Lower the leg slowly back to the bent starting position. Avoid locking the knee aggressively.",
  ],

  // ── Hip ──────────────────────────────────────────────────────
  "hip.flexion": [
    "Stand about 6 feet (2 metres) from the camera, turned so the camera sees your side. Hold a wall or chair lightly for balance if needed.",
    "Stand tall with both legs straight. The camera should see your shoulder, hip, knee, and ankle on the test side.",
    "Slowly lift the test-side knee forward and upward, bringing the thigh toward your chest. Keep the standing leg straight and trunk upright.",
    "Hold the highest comfortable position for 2 seconds. Don't lean back to gain extra range.",
    "Slowly lower the leg back to standing. Repeat on the other side if assessing both.",
  ],
  "hip.extension": [
    "Stand about 6 feet (2 metres) from the camera, turned so the camera sees your side. Use a wall or chair for balance if needed.",
    "Stand tall with both legs straight. Keep the trunk upright throughout.",
    "Slowly move the test-side leg backward, keeping the knee straight. The thigh should travel behind the body line.",
    "Hold the furthest backward position for 2 seconds. Important: don't arch your lower back — that fakes the movement.",
    "Slowly bring the leg back to standing. Repeat on the other side if assessing both.",
  ],
  "hip.internal_rotation": [
    "Sit on a chair facing the camera, with the camera showing both legs from the front. Both feet flat on the floor.",
    "Keep the knee at 90° and the thigh still. The hip rotation happens at the hip, not the knee.",
    "Slowly rotate the lower leg outward (so the foot moves AWAY from the midline) — this is internal rotation at the hip joint.",
    "Hold the furthest comfortable position for 2 seconds. Keep the thigh and pelvis still on the chair.",
    "Slowly bring the foot back to straight. Note: 2D camera measurement of rotation is approximate.",
  ],
  "hip.external_rotation": [
    "Sit on a chair facing the camera. Both feet flat on the floor, knees at 90°.",
    "Keep the knee at 90° and the thigh still. The rotation must come from the hip, not by tilting the knee.",
    "Slowly rotate the lower leg inward (so the foot crosses TOWARD the midline / opposite leg) — this is external rotation at the hip joint.",
    "Hold the furthest comfortable position for 2 seconds. Keep your pelvis still on the chair seat.",
    "Slowly return the foot to straight. Note: 2D camera measurement of rotation is approximate.",
  ],

  // ── Ankle ────────────────────────────────────────────────────
  "ankle.flexion": [
    "Stand about 4 feet (1.2 metres) from the camera, turned so the camera sees the side of the leg you're testing.",
    "Place the test foot flat on the floor with the toe a fixed distance from a wall (or imaginary line). Keep the heel firmly on the ground throughout.",
    "Slowly bend the knee forward, trying to make the knee touch the wall while keeping your heel on the floor. The shin will tilt forward over the foot.",
    "Hold the deepest comfortable lean for 2 seconds. Don't lift the heel off the ground at any point.",
    "Slowly straighten back to standing. This is the standard knee-to-wall dorsiflexion test.",
  ],
  "ankle.extension": [
    "Sit on a chair with one leg extended out, turned so the camera sees the side of that leg. Use a stool or another chair to support the leg if needed.",
    "Keep the leg relaxed and straight. The camera should clearly see the knee and ankle.",
    "Slowly point the foot downward as far as you can — like pressing a gas pedal. Move only at the ankle, keeping the leg still.",
    "Hold the most-pointed position for 2 seconds.",
    "Slowly bring the foot back to neutral (90° to the leg). Repeat on the other side if assessing both.",
  ],
};

export function getInstructions(
  bodyPart: "shoulder" | "neck" | "knee" | "hip" | "ankle",
  movementId: string,
): string[] {
  return MOVEMENT_INSTRUCTIONS[`${bodyPart}.${movementId}` as Key] ?? [];
}

export function isRotationMovement(
  bodyPart: "shoulder" | "neck" | "knee" | "hip" | "ankle",
  movementId: string,
): boolean {
  if (bodyPart === "shoulder" || bodyPart === "hip") {
    return movementId === "external_rotation" || movementId === "internal_rotation";
  }
  if (bodyPart === "neck") return movementId === "rotation";
  return false;
}
