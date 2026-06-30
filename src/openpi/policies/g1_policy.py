import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Unitree G1 (Dex1 bimanual) action/state layout (16 dims):
#   0..6   left arm joints  (ShoulderPitch, ShoulderRoll, ShoulderYaw, Elbow, WristRoll, WristPitch, WristYaw)
#   7..13  right arm joints (same order)
#   14     left gripper
#   15     right gripper
G1_ACTION_DIM = 16


def make_g1_example() -> dict:
    """Creates a random input example for the Unitree G1 policy (matches the inference key format)."""
    return {
        "observation/state": np.random.rand(G1_ACTION_DIM),
        "observation/cam_left_high": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/cam_left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/cam_right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "put the battery into the battery bin and the screw driver into the Philips bin",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):
    """Converts Unitree G1 observations into the format expected by the model.

    Used for both training and inference. The keys read here must match the keys produced by the
    repack transform (training) and the keys sent by the inference client (deployment).
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # LeRobot stores video frames as float32 (C,H,W); _parse_image normalizes to uint8 (H,W,C).
        # The G1 has a head/third-person camera (cam_left_high) plus two wrist cameras -- all three are
        # real, so all image masks are True.
        base_image = _parse_image(data["observation/cam_left_high"])
        left_wrist = _parse_image(data["observation/cam_left_wrist"])
        right_wrist = _parse_image(data["observation/cam_right_wrist"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Language instruction.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):
    """Converts model outputs back to the Unitree G1 action space (inference only)."""

    def __call__(self, data: dict) -> dict:
        # Model actions are padded to the model action dim; return only the 16 real G1 dims.
        return {"actions": np.asarray(data["actions"][..., :G1_ACTION_DIM])}
