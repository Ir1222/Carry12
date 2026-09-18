"""Physical carry diagnostic schema, with no simulator or analysis dependency."""

TEMPORAL_METRICS = (
    "left_hand_tangential_slip_mps", "right_hand_tangential_slip_mps",
    "left_hand_slip_violation_mps", "right_hand_slip_violation_mps",
    "box_relative_motion_error_mps", "box_relative_velocity_violation_mps",
    "box_relative_velocity_x_mps", "box_relative_velocity_y_mps", "box_relative_velocity_z_mps",
)
METRIC_NAMES = (
    "left_hand_side_error_m", "right_hand_side_error_m",
    "left_hand_side_violation_m", "right_hand_side_violation_m",
    "left_hand_face_overflow_m", "right_hand_face_overflow_m",
    "left_hand_face_violation_m", "right_hand_face_violation_m",
    "arm_range_violation_rad", "arm_range_clearance_rad",
    "box_relative_region_violation_m", "box_relative_region_clearance_m",
    "box_relative_position_x_m", "box_relative_position_y_m", "box_relative_position_z_m",
    "bilateral_contact_rate",
) + TEMPORAL_METRICS
