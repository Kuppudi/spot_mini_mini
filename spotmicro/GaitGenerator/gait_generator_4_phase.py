import copy
import numpy as np

from spotmicro.GaitGenerator.Bezier import BezierGait, STANCE, SWING
from spotmicro.Kinematics.LieAlgebra import TransToRp


class BezierGait4Phase(BezierGait):
    """One-leg-at-a-time four-phase gait using the existing Bezier foot arcs.

    The original Bezier gait generator in this repo hardcodes trot-style
    offsets inside GenerateTrajectory(). This class keeps the same Bezier
    swing/stance curves but replaces the phase scheduler so each leg can move
    in its own quarter of the gait cycle.
    """

    # Use a diagonal crawl sequence instead of front-front-back-back.
    # This keeps the support polygon shifting more evenly and avoids
    # visibly favoring one front leg.
    LEG_SEQUENCE = ("FL", "BR", "FR", "BL")
    LEG_ORDER = ("FL", "FR", "BL", "BR")

    def __init__(
        self,
        leg_sequence=LEG_SEQUENCE,
        dt=0.01,
        Tswing=0.16,
        max_swing_ratio=0.18,
        inter_swing_stance_ratio=0.08,
        preload_ratio=0.18,
        preload_step_scale=0.26,
        preload_clearance_scale=0.08,
    ):
        super().__init__(dSref=[0.0, 0.25, 0.5, 0.75], dt=dt, Tswing=Tswing)

        if sorted(leg_sequence) != sorted(self.LEG_ORDER):
            raise ValueError(
                "leg_sequence must contain each of FL, FR, BL, BR exactly once."
            )

        self.leg_sequence = tuple(leg_sequence)
        self.phase_offsets = {
            leg: index / float(len(self.leg_sequence))
            for index, leg in enumerate(self.leg_sequence)
        }
        self.max_swing_ratio = max_swing_ratio
        self.inter_swing_stance_ratio = inter_swing_stance_ratio
        self.preload_ratio = preload_ratio
        self.preload_step_scale = preload_step_scale
        self.preload_clearance_scale = preload_clearance_scale
        # Keep physical indices in canonical FL, FR, BL, BR order so
        # downstream geometry like YawCircle() still identifies each leg
        # correctly even when the phase sequence changes.
        self.leg_index_map = {leg: index for index, leg in enumerate(self.LEG_ORDER)}

    def reset(self):
        super().reset()
        self.Phases = [0.0, 0.0, 0.0, 0.0]

    def _get_cycle_timing(self, step_length, step_velocity, dt):
        representative_step_length = step_length
        if isinstance(step_length, dict):
            representative_step_length = max(
                abs(float(value)) for value in step_length.values()
            )
        elif np.ndim(step_length) > 0:
            representative_step_length = float(np.max(np.abs(np.asarray(step_length, dtype=float))))

        if abs(step_velocity) < 1e-6 or abs(representative_step_length) < 1e-6:
            return None

        desired_stance = 2.0 * abs(representative_step_length) / abs(step_velocity)
        desired_cycle = desired_stance + self.Tswing
        min_cycle = self.Tswing / self.max_swing_ratio
        cycle_duration = max(desired_cycle, min_cycle, self.Tswing + dt)
        phase_spacing = 1.0 / float(len(self.leg_sequence))
        max_ratio_with_gap = max(0.05, phase_spacing - self.inter_swing_stance_ratio)
        swing_ratio = min(
            self.Tswing / cycle_duration,
            self.max_swing_ratio,
            max_ratio_with_gap,
        )
        return cycle_duration, swing_ratio

    def _resolve_leg_value(self, value, leg_name, leg_index):
        if isinstance(value, dict):
            return float(value.get(leg_name, 0.0))
        if np.ndim(value) > 0:
            value_array = np.asarray(value, dtype=float).reshape(-1)
            if leg_index < value_array.size:
                return float(value_array[leg_index])
            return 0.0
        return float(value)

    def _get_leg_phase(self, leg_name, cycle_duration, swing_ratio):
        global_phase = (self.time % cycle_duration) / cycle_duration
        leg_phase = (global_phase - self.phase_offsets[leg_name]) % 1.0
        leg_index = self.leg_index_map[leg_name]

        if leg_phase < swing_ratio:
            swing_phase = leg_phase / max(swing_ratio, 1e-6)
            self.Phases[leg_index] = 1.0 + swing_phase
            return swing_phase, SWING

        stance_phase = (leg_phase - swing_ratio) / max(1.0 - swing_ratio, 1e-6)
        self.Phases[leg_index] = stance_phase
        return stance_phase, STANCE

    def _get_preload_progress(self, leg_name, cycle_duration, swing_ratio):
        if self.preload_ratio <= 1e-6:
            return 0.0

        global_phase = (self.time % cycle_duration) / cycle_duration
        leg_phase = (global_phase - self.phase_offsets[leg_name]) % 1.0

        # Only preload in the final slice of stance, right before the
        # leg enters swing on the next cycle.
        preload_start = max(swing_ratio, 1.0 - self.preload_ratio)
        if leg_phase < preload_start or leg_phase < swing_ratio:
            return 0.0

        progress = (leg_phase - preload_start) / max(1.0 - preload_start, 1e-6)
        return float(np.clip(progress, 0.0, 1.0))

    def _apply_preload(self, coord, L, lateral_fraction, clearance_height, preload_progress):
        if preload_progress <= 0.0:
            return coord

        preload_blend = np.sin(0.5 * np.pi * preload_progress) ** 2
        forward_dir = np.cos(lateral_fraction)
        lateral_dir = np.sin(lateral_fraction)
        step_sign = 1.0 if L >= 0.0 else -1.0
        preload_step = self.preload_step_scale * abs(L) * preload_blend
        # Keep the foot mostly loaded through preload so the body stays level.
        # We only allow a tiny amount of late unweighting right before swing.
        lift_progress = np.clip((preload_progress - 0.70) / 0.30, 0.0, 1.0)
        preload_lift = (
            self.preload_clearance_scale
            * clearance_height
            * (lift_progress ** 2)
        )

        coord = coord.copy()
        coord[0] += step_sign * preload_step * forward_dir
        coord[1] += step_sign * preload_step * lateral_dir
        coord[2] += preload_lift
        return coord

    def GenerateTrajectory(
        self,
        L,
        LateralFraction,
        YawRate,
        vel,
        T_bf_,
        T_bf_curr,
        clearance_height=0.06,
        penetration_depth=0.01,
        contacts=None,
        dt=None,
    ):
        if dt is None:
            dt = self.dt

        YawRate *= dt
        timing = self._get_cycle_timing(L, vel, dt)

        if timing is None:
            self.reset()
            return copy.deepcopy(T_bf_)

        cycle_duration, swing_ratio = timing
        T_bf = copy.deepcopy(T_bf_)

        for leg_name, Tbf_in in T_bf_.items():
            leg_index = self.leg_index_map[leg_name]
            _, p_bf = TransToRp(Tbf_in)
            leg_step_length = self._resolve_leg_value(L, leg_name, leg_index)
            phase, stance_swing = self._get_leg_phase(
                leg_name, cycle_duration, swing_ratio
            )

            if stance_swing == STANCE:
                step_coord = self.StanceStep(
                    phase,
                    leg_step_length,
                    LateralFraction,
                    YawRate,
                    penetration_depth,
                    p_bf,
                    leg_name,
                    leg_index,
                )
                preload_progress = self._get_preload_progress(
                    leg_name, cycle_duration, swing_ratio
                )
                step_coord = self._apply_preload(
                    step_coord,
                    leg_step_length,
                    LateralFraction,
                    clearance_height,
                    preload_progress,
                )
            else:
                step_coord = self.SwingStep(
                    phase,
                    leg_step_length,
                    LateralFraction,
                    YawRate,
                    clearance_height,
                    p_bf,
                    leg_name,
                    leg_index,
                )

            T_bf[leg_name][0, 3] = Tbf_in[0, 3] + step_coord[0]
            T_bf[leg_name][1, 3] = Tbf_in[1, 3] + step_coord[1]
            T_bf[leg_name][2, 3] = Tbf_in[2, 3] + step_coord[2]

        self.time = (self.time + dt) % cycle_duration
        return T_bf
