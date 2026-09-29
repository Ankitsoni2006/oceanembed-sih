/**
 * Shared display constants for the North Indian Ocean dashboard.
 *
 * The thermocline band is a fixed, nominal display band used to shade charts and
 * tables consistently in Reconstruction and ARGO Validation modes. It is NOT a
 * per-profile diagnosis: the profile-specific thermocline depth (max |dT/dz|) is
 * computed by the backend and shown in Reconstruction Mode.
 */
export const THERMOCLINE_TOP_M = 75;
export const THERMOCLINE_BOTTOM_M = 150;

export const isInThermoclineBand = (depthM: number): boolean =>
  depthM >= THERMOCLINE_TOP_M && depthM <= THERMOCLINE_BOTTOM_M;
