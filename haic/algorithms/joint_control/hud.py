"""Small, stateless decoders for the public 84x84 grayscale HUD.

Coordinates and signs come from car_racing.py's indicator rectangles. Speed is
the *norm* of hull velocity, wheel_angle is front joint[0].angle (not the front
mean or the steering command), and yaw_rate is physical CCW angular velocity.
In particular, positive action steering targets a negative physical joint angle.

Grayscale loses color identity and the two resizes lose subpixel information.
Confidence below is a structural image-quality score, NOT a calibrated error
probability. Source defaults are not a substitute for held-geometry validation.
No simulator, controller, privileged state, or temporal/future input is used.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


HUD_CHANNELS = (
    "speed", "wheel_angle", "yaw_rate", "wheel_omega_0", "wheel_omega_1",
    "wheel_omega_2", "wheel_omega_3",
)


@dataclass(frozen=True)
class HudCalibration:
    """Independent affine corrections in HUD_CHANNELS order.

    Zero samples means a source-only default, not an empirically fitted channel.
    Counts describe supplied valid FIT rows, not independent trials or accuracy.
    Geometry splitting, deduplication, and label provenance belong to the caller.
    """

    gain: Sequence[float] = (1.0,) * 7
    bias: Sequence[float] = (0.0,) * 7
    fit_samples: Sequence[int] = (0,) * 7

    def __post_init__(self):
        for name in ("gain", "bias", "fit_samples"):
            values = np.asarray(getattr(self, name))
            if values.shape != (7,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain seven finite values")
        if np.any(np.asarray(self.gain) <= 0):
            raise ValueError("calibration gains must preserve physical signs")
        counts = np.asarray(self.fit_samples)
        if np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError("fit_samples must contain nonnegative integers")
        object.__setattr__(self, "gain", tuple(float(x) for x in self.gain))
        object.__setattr__(self, "bias", tuple(float(x) for x in self.bias))
        object.__setattr__(self, "fit_samples", tuple(int(x) for x in self.fit_samples))


def _resized_interval(left, right, source_size):
    """Area shrink to 96, then pixel-center linear resize to 84, in one axis."""
    edges = np.arange(97) * (source_size / 96.0)
    coverage = np.maximum(0.0, np.minimum(edges[1:], right) -
                          np.maximum(edges[:-1], left)) / (source_size / 96.0)
    centers = (np.arange(84) + 0.5) * (96.0 / 84.0) - 0.5
    return np.interp(centers, np.arange(96), coverage)


# Four fixed spatial kernels, not learned image features. The one-source-pixel
# inclusive polygon edge is retained; integer/color rounding remains approximate.
_WHEEL_KERNEL = np.stack([
    _resized_interval(175 + 25 * i, 201 + 25 * i, 1000)[14:25]
    * ((29 if i < 2 else 44) / 255.0) for i in range(4)
], axis=1)
_WHEEL_INVERSE = np.linalg.pinv(_WHEEL_KERNEL).T
_WHEEL_BASE = _resized_interval(780, 781, 800)[74:84]
_WHEEL_BASE_CENTER = float(np.dot(_WHEEL_BASE, np.arange(74, 84)) /
                           _WHEEL_BASE.sum())


def extract_hud_features(frames) -> dict[str, np.ndarray]:
    """Extract fixed low-dimensional features from (84,84) or (N,84,84).

    Accept uint8 or finite floating point values already normalized to [0,1].
    A four-frame observation is a batch of four independent frames, not an
    instruction to average them. Every returned array preserves the batch axis;
    single frames remove that axis. Empty batches are supported.

    raw_values, confidence, valid and zero_signal have final dimension seven in
    HUD_CHANNELS order. Other fields expose area/quality diagnostics. Invalid
    raw values are retained for diagnosis; decode_hud masks them with NaN.
    """
    pixels = np.asarray(frames)
    single = pixels.ndim == 2
    if pixels.ndim not in (2, 3) or pixels.shape[-2:] != (84, 84):
        raise ValueError("frames must have shape (84, 84) or (N, 84, 84)")
    if pixels.dtype != np.uint8:
        if not np.issubdtype(pixels.dtype, np.floating):
            raise ValueError("frames must be uint8 or normalized floating point")
        if not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 1)):
            raise ValueError("floating point frames must be finite and in [0, 1]")
    if single:
        pixels = pixels[None]
    hud = pixels[:, 74:84].astype(np.float64)
    if pixels.dtype == np.uint8:
        hud /= 255.0
    n = len(hud)
    raw = np.zeros((n, 7))
    confidence = np.zeros((n, 7))
    valid = np.zeros((n, 7), dtype=bool)
    zero = np.zeros((n, 7), dtype=bool)
    background = hud[:, 8:10, 28:80].max(axis=(1, 2))
    # Reward text can witness a stationary HUD, but its numeric value is unused.
    present = (hud[:, 1:9].sum(axis=(1, 2)) > 2 / 255) & (background < 2 / 255)

    speed_columns = hud[:, :, 10:14].sum(axis=1)
    speed_mass = speed_columns.sum(axis=1)
    # 0.042 px/unit/s * 2.1 px width; mean inclusive/quantized source-edge bias.
    raw[:, 0] = np.maximum(0.0, (speed_mass - 1.5 * .105 * 2.1) / (.042 * 2.1))
    zero[:, 0] = speed_mass == 0
    expected = np.array([.5, 1.0, .6, 1 / 30])
    expected /= expected.sum()
    speed_shape_error = np.abs(speed_columns / np.maximum(speed_mass[:, None],
                                                        1e-12) - expected).sum(axis=1)
    speed_shape_error[zero[:, 0]] = 0
    valid[:, 0] = present & (hud[:, 0, 10:14].max(axis=1) < 2 / 255) & (speed_shape_error < .2)
    confidence[:, 0] = np.clip(1 - speed_shape_error / .2, 0, 1)

    profile = hud[:, 3:5].mean(axis=1)  # y77:79 is the horizontal-bar interior.
    horizontal_mass = np.zeros((n, 2, 2))
    gap_active = profile[:, 54:56].max(axis=1) > 2 / 255
    for channel, left, anchor, right, level, scale, raw_scale in (
        (1, 29, 42, 55, 149 / 255, 21.0, 250.0),
        (2, 55, 63, 84, 76 / 255, 1.68, 20.0),
    ):
        crop = profile[:, left:right]
        mass_left = profile[:, left:anchor].sum(axis=1)
        mass_right = profile[:, anchor:right].sum(axis=1)
        mass = mass_left + mass_right
        horizontal_mass[:, channel - 1] = np.stack((mass_left, mass_right), axis=1)
        zero[:, channel] = mass == 0
        # Both bars point LEFT for positive physical values. Remove the shared
        # anchor pixel, then use the midpoint of the raw-pixel quantization bin.
        raw[:, channel] = ((mass_left - mass_right) / level + .084) / scale - .5 / raw_scale
        raw[zero[:, channel], channel] = 0
        both_sides = (mass_left > .2 * level) & (mass_right > .2 * level)
        edge = (crop[:, 0] > 2 / 255) | (crop[:, -1] > 2 / 255)
        too_bright = crop.max(axis=1) > level + 2 / 255
        # A disconnected component cannot be an anchored rectangle. Permit its
        # low-intensity antialias edge but not a dark hole toward the anchor.
        active = crop > 2 / 255
        indices = np.arange(right - left)
        first = np.where(active, indices, right - left).min(axis=1)
        last = np.where(active, indices, -1).max(axis=1)
        span = (indices[None] >= np.minimum(first, anchor - left)[:, None]) & (
            indices[None] <= np.maximum(last, anchor - left)[:, None])
        hole = np.any(span & ~active, axis=1) & ~zero[:, channel]
        valid[:, channel] = present & ~edge & ~too_bright & ~both_sides & ~hole & ~gap_active
        # Tiny bars have a coarse relative/sign resolution even if well formed.
        confidence[:, channel] = np.where(zero[:, channel], .75,
                                          .35 + .6 * np.minimum(mass / level, 1))

    # Optional wheel bars are adjacent and dim after grayscale. Unmix only their
    # fixed horizontal footprints, then recover vertical extent and direction.
    wheel_crop = hud[:, :, 14:25]
    wheel_profiles = wheel_crop @ _WHEEL_INVERSE
    reconstructed = wheel_profiles @ _WHEEL_KERNEL.T
    wheel_residual = np.abs(reconstructed - wheel_crop).sum(axis=(1, 2)) / np.maximum(
        wheel_crop.sum(axis=(1, 2)), .1)
    wheel_profiles = np.maximum(wheel_profiles, 0)
    wheel_mass = wheel_profiles.sum(axis=1)
    moment = np.sum(wheel_profiles * (np.arange(74, 84) - _WHEEL_BASE_CENTER)[None, :, None], axis=1)
    direction = np.where(moment < -1e-4, 1.0, -1.0)
    raw[:, 3:] = direction * np.maximum(0, wheel_mass - _WHEEL_BASE.sum()) / .021 - 2.5
    zero[:, 3:] = wheel_mass < .03
    raw[:, 3:][zero[:, 3:]] = 0
    wheel_clipped = (wheel_profiles[:, 0] > .04) | (wheel_profiles[:, -1] > .04)
    # Short bars on either side of y780 can occupy the SAME 96px resize bin.
    # Their sign is genuinely ambiguous, not resolved by fitting a gain/bias.
    wheel_direction_supported = zero[:, 3:] | (np.abs(moment) > .1)
    valid[:, 3:] = (present[:, None] & ~wheel_clipped & (wheel_residual[:, None] < .25)
                   & (wheel_profiles.max(axis=1) < 1.25) & wheel_direction_supported)
    confidence[:, 3:] = .6 * np.clip(1 - wheel_residual[:, None] / .25, 0, 1)
    raw[zero] = 0
    confidence[~valid] = 0
    features = {
        "raw_values": raw, "confidence": confidence, "valid": valid,
        "zero_signal": zero, "speed_mass": speed_mass,
        "speed_shape_error": speed_shape_error, "horizontal_mass": horizontal_mass,
        "wheel_mass": wheel_mass, "wheel_residual": wheel_residual,
        "hud_present": present,
    }
    return {key: np.asarray(value[0] if single else value) for key, value in features.items()}


def fit_hud_calibration(features: Mapping[str, np.ndarray],
                        labels: Mapping[str, np.ndarray]) -> HudCalibration:
    """Fit seven or fewer 1-D affine maps on a caller-owned FIT subset ONLY.

    The caller must split whole geometries and deduplicate before calling. This
    function has no file access, implicit split, holdout selection, or searching.
    Labels may contain speed, wheel_angle, yaw_rate, wheel_omega (last dim four).
    Omitted channels retain source defaults; nonfinite targets and structurally
    invalid observations are excluded and exact counts returned. Do NOT pass
    action-derived joint estimates or heading differences as instantaneous truth.
    """
    allowed = {"speed", "wheel_angle", "yaw_rate", "wheel_omega"}
    if not labels or not set(labels) <= allowed:
        raise ValueError("labels must contain only named physical HUD channels")
    raw = np.asarray(features["raw_values"], dtype=np.float64)
    valid = np.asarray(features["valid"], dtype=bool)
    if raw.ndim != 2 or raw.shape[1] != 7 or valid.shape != raw.shape:
        raise ValueError("calibration requires a batch of (N, 7) features")
    gain, bias, counts = np.ones(7), np.zeros(7), np.zeros(7, dtype=int)
    for key, target in labels.items():
        start = ("speed", "wheel_angle", "yaw_rate", "wheel_omega").index(key)
        width = 4 if key == "wheel_omega" else 1
        target = np.asarray(target, dtype=np.float64)
        expected = (len(raw), 4) if width == 4 else (len(raw),)
        if target.shape != expected:
            raise ValueError(f"{key} labels must have shape {expected}")
        target = target.reshape(len(raw), width)
        if key == "speed" and np.any(target[np.isfinite(target)] < 0):
            raise ValueError("speed labels are nonnegative velocity norms")
        for offset in range(width):
            channel = start + offset
            mask = valid[:, channel] & np.isfinite(raw[:, channel]) & np.isfinite(target[:, offset])
            x, y = raw[mask, channel], target[mask, offset]
            if len(x) < 8 or np.ptp(x) < 1e-6:
                raise ValueError(f"insufficient varying FIT data for {HUD_CHANNELS[channel]}")
            slope, intercept = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y,
                                             rcond=None)[0]
            if not np.isfinite([slope, intercept]).all() or not .25 <= slope <= 4:
                raise ValueError(f"implausible calibration/sign for {HUD_CHANNELS[channel]}")
            gain[channel], bias[channel], counts[channel] = slope, intercept, len(x)
    return HudCalibration(tuple(gain), tuple(bias), tuple(int(x) for x in counts))


def decode_hud(frames, calibration: HudCalibration | None = None) -> dict[str, np.ndarray]:
    """Decode physical scalars plus (...,7) confidence/valid masks.

    Invalid channels are NaN, never silently clipped to plausible physics. Zero
    visible area means zero only on a structurally present HUD. Confidence is
    image quality, not uncertainty coverage; optional wheel bars are lower quality.
    Short wheel bars near the baseline abstain because up/down signs can alias.
    """
    calibration = HudCalibration() if calibration is None else calibration
    features = extract_hud_features(frames)
    values = (features["raw_values"] * np.asarray(calibration.gain)
              + np.asarray(calibration.bias))
    values = np.where(features["zero_signal"], 0, values)
    values[..., 0] = np.maximum(0, values[..., 0])
    values = np.where(features["valid"], values, np.nan)
    return {
        "speed": values[..., 0], "wheel_angle": values[..., 1],
        "yaw_rate": values[..., 2], "wheel_omega": values[..., 3:],
        "confidence": features["confidence"], "valid": features["valid"],
    }
