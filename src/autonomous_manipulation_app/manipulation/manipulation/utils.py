# ===== BASE FRAME NAMES =====

BODY_FRAME = "body"
ODOM_FRAME = "odom"
VISION_FRAME = "vision"
HAND_FRAME = "hand"
HEAD_FRAME = "head"

# ===== INTERMEDIATE CAMERA FRAME NAMES =====

FRONTRIGHT_FRAME = "frontright"
FRONTLEFT_FRAME = "frontleft"
RIGHT_FRAME = "right"
LEFT_FRAME = "left"
BACK_FRAME = "back"


# ===== CAMERA FRAME NAMES =====

BACK_CAMERA_FRAME = "back_fisheye"
FRONTLEFT_CAMERA_FRAME = "frontleft_fisheye"
FRONTRIGHT_CAMERA_FRAME = "frontright_fisheye"
LEFT_CAMERA_FRAME = "left_fisheye"
RIGHT_CAMERA_FRAME = "right_fisheye"
HAND_CAMERA_FRAME = "hand_color_image_sensor"


# ===== ARM FRAME NAMES =====

ARM_SH0_FRAME = "arm_link_sh0"
ARM_SH1_FRAME = "arm_link_sh1"
ARM_HR0_FRAME = "arm_link_hr0"
ARM_EL0_FRAME = "arm_link_el0"
ARM_EL1_FRAME = "arm_link_el1"
ARM_WR0_FRAME = "arm_link_wr0"
ARM_WR1_FRAME = "arm_link_wr1"


# ===== BODY / HEAD CONNECTION =====

BODY_HEAD_EDGES = (
    (BODY_FRAME, HEAD_FRAME),
)

# ===== HEAD CAMERA BRANCHES =====
# Each chain starts at head and ends at a camera.

HEAD_CAMERA_CHAINS = (
    (HEAD_FRAME, FRONTRIGHT_FRAME, FRONTRIGHT_CAMERA_FRAME),
    (HEAD_FRAME, FRONTLEFT_FRAME, FRONTLEFT_CAMERA_FRAME),
    (HEAD_FRAME, RIGHT_FRAME, RIGHT_CAMERA_FRAME),
    (HEAD_FRAME, LEFT_FRAME, LEFT_CAMERA_FRAME),
    (HEAD_FRAME, BACK_FRAME, BACK_CAMERA_FRAME),
)

HEAD_CAMERA_EDGES = tuple(
    (parent, child)
    for chain in HEAD_CAMERA_CHAINS
    for parent, child in zip(chain[:-1], chain[1:])
)

# ===== STATIC TRANSFORM EDGES =====
# Same connections as in your original static_edges list.

STATIC_EDGES = BODY_HEAD_EDGES + HEAD_CAMERA_EDGES


# ===== ARM / HAND CAMERA CHAIN =====

ARM_CAMERA_CHAIN = (
    ARM_SH0_FRAME,
    ARM_SH1_FRAME,
    ARM_HR0_FRAME,
    ARM_EL0_FRAME,
    ARM_EL1_FRAME,
    ARM_WR0_FRAME,
    ARM_WR1_FRAME,
    HAND_CAMERA_FRAME,
)

ARM_CAMERA_EDGES = tuple(
    zip(ARM_CAMERA_CHAIN[:-1], ARM_CAMERA_CHAIN[1:])
)

# ===== ODOM TO HAND CAMERA CHAIN =====
# Includes both static and dynamic connections.

ODOM_TO_HAND_CAMERA_CHAIN = (
    ODOM_FRAME,
    BODY_FRAME,
    ARM_SH0_FRAME,
    ARM_SH1_FRAME,
    ARM_HR0_FRAME,
    ARM_EL0_FRAME,
    ARM_EL1_FRAME,
    ARM_WR0_FRAME,
    ARM_WR1_FRAME,
    HAND_CAMERA_FRAME,
)

# Consecutive (parent, child) pairs.
ODOM_TO_HAND_CAMERA_EDGES = tuple(
    zip(
        ODOM_TO_HAND_CAMERA_CHAIN[:-1],
        ODOM_TO_HAND_CAMERA_CHAIN[1:],
    )
)


# ===== CAMERA INTRINSICS =====

CAMERA_INTRINSICS = {
    FRONTLEFT_CAMERA_FRAME: {
        'fx': 330.7691955566406,
        'fy': 330.9757080078125,
        'cx': 318.3247375488281,
        'cy': 237.50799560546875
    },
    FRONTRIGHT_CAMERA_FRAME: {
        'fx': 329.4346008300781,
        'fy': 329.8037414550781,
        'cx': 314.97747802734375,
        'cy': 237.4246063232422
    },
    LEFT_CAMERA_FRAME: {
        'fx': 329.6697692871094,
        'fy': 329.885009765625,
        'cx': 319.45611572265625,
        'cy': 235.0216827392578
    },
    RIGHT_CAMERA_FRAME: {
        'fx': 331.18707275390625,
        'fy': 331.398681640625,
        'cx': 309.7239074707031,
        'cy': 242.5687713623047
    },
    BACK_CAMERA_FRAME: {
        'fx': 329.2738037109375,
        'fy': 329.7651672363281,
        'cx': 327.1336975097656,
        'cy': 238.9302215576172
    },
    HAND_CAMERA_FRAME: {
        'fx': 552.0291012161067,
        'fy': 552.0291012161067,
        'cx': 320.0,
        'cy': 240.0
    },
}