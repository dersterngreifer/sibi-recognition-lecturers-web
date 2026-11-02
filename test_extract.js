const { extractKeypoints } = require("./lib/mediapipeUtils.js") || {};
// We can't easily require ts directly without transpiling, let's just copy the function here to test it

const KEYPOINTS_CONFIG = {
  POSE_KEYPOINTS: 33,
  POSE_DIMENSIONS: 4,
  LEFT_HAND_KEYPOINTS: 21,
  RIGHT_HAND_KEYPOINTS: 21,
  HAND_DIMENSIONS: 3,
  TOTAL_FEATURES: 258,
};

function fillVisibilityFallbackIfEmpty(keypoints) {}

function extractKeypointsTest(detectionResult, mirrored = false) {
  const total = KEYPOINTS_CONFIG.TOTAL_FEATURES;
  const keypoints = new Float32Array(total);
  let i = 0;

  const flipX = (v) => (mirrored ? 1 - v : v);

  if (detectionResult.poseLandmarks && KEYPOINTS_CONFIG.POSE_KEYPOINTS) {
    for (let p = 0; p < KEYPOINTS_CONFIG.POSE_KEYPOINTS; p++) {
      const lm = detectionResult.poseLandmarks[p];
      keypoints[i++] = lm ? flipX(lm.x) : 0;
      keypoints[i++] = lm?.y ?? 0;
      keypoints[i++] = lm?.z ?? 0;
      keypoints[i++] = lm?.visibility ?? 0;
    }
  } else {
    i += (KEYPOINTS_CONFIG.POSE_KEYPOINTS || 33) * KEYPOINTS_CONFIG.POSE_DIMENSIONS;
  }

  const leftHandLandmarks = mirrored
    ? detectionResult.rightHandLandmarks
    : detectionResult.leftHandLandmarks;
  const rightHandLandmarks = mirrored
    ? detectionResult.leftHandLandmarks
    : detectionResult.rightHandLandmarks;

  if (leftHandLandmarks && KEYPOINTS_CONFIG.LEFT_HAND_KEYPOINTS) {
    for (let p = 0; p < KEYPOINTS_CONFIG.LEFT_HAND_KEYPOINTS; p++) {
      const lm = leftHandLandmarks[p];
      keypoints[i++] = lm ? flipX(lm.x) : 0;
      keypoints[i++] = lm?.y ?? 0;
      keypoints[i++] = lm?.z ?? 0;
    }
  } else {
    i += (KEYPOINTS_CONFIG.LEFT_HAND_KEYPOINTS || 21) * KEYPOINTS_CONFIG.HAND_DIMENSIONS;
  }

  if (rightHandLandmarks && KEYPOINTS_CONFIG.RIGHT_HAND_KEYPOINTS) {
    for (let p = 0; p < KEYPOINTS_CONFIG.RIGHT_HAND_KEYPOINTS; p++) {
      const lm = rightHandLandmarks[p];
      keypoints[i++] = lm ? flipX(lm.x) : 0;
      keypoints[i++] = lm?.y ?? 0;
      keypoints[i++] = lm?.z ?? 0;
    }
  } else {
    i += (KEYPOINTS_CONFIG.RIGHT_HAND_KEYPOINTS || 21) * KEYPOINTS_CONFIG.HAND_DIMENSIONS;
  }

  fillVisibilityFallbackIfEmpty(keypoints);
  return { keypoints, final_i: i };
}

// Test with empty
console.log("Empty:", extractKeypointsTest({}).final_i);

// Test with pose only
console.log("Pose only:", extractKeypointsTest({
  poseLandmarks: Array(33).fill({x:0, y:0, z:0, visibility:0})
}).final_i);

// Test with right hand only (mirrored)
console.log("Right hand only (mirrored=true):", extractKeypointsTest({
  rightHandLandmarks: Array(21).fill({x:0, y:0, z:0})
}, true).final_i);

