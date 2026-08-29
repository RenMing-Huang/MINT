"""Video and trajectory output for transfer rollouts."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np


class VideoWriter:
    def __init__(self, path: str | Path, fps: int = 30):
        import imageio.v2 as imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(self.path, fps=fps)

    def append(self, frame: np.ndarray) -> None:
        self._writer.append_data(np.asarray(frame, dtype=np.uint8))

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _trajectory_markers(
    reference: np.ndarray,
    executed: np.ndarray,
    stride: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build time-synchronized green reference and blue execution markers."""
    if stride < 1:
        raise ValueError("trajectory marker stride must be positive")

    markers: list[tuple[np.ndarray, np.ndarray]] = []

    def add(values: np.ndarray, start_rgb: np.ndarray, end_rgb: np.ndarray) -> None:
        sampled = np.asarray(values, dtype=np.float64)[::stride]
        for index, position in enumerate(sampled):
            amount = index / max(1, len(sampled) - 1)
            rgb = start_rgb + amount * (end_rgb - start_rgb)
            markers.append((position, np.append(rgb, 0.85)))

    add(
        reference[: len(executed)],
        np.array([0.55, 0.90, 0.60]),
        np.array([0.05, 0.45, 0.15]),
    )
    add(
        executed,
        np.array([0.55, 0.75, 1.00]),
        np.array([0.05, 0.25, 0.85]),
    )
    return markers


def _add_markers_to_scene(render_context, markers) -> None:
    import mujoco

    for position, rgba in markers:
        if render_context.scn.ngeom >= render_context.scn.maxgeom:
            break
        geom = render_context.scn.geoms[render_context.scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.005, 0.005, 0.005]),
            np.asarray(position, dtype=np.float64),
            np.eye(3).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.emission = 0.5
        render_context.scn.ngeom += 1


def _render_with_markers(
    render_context,
    width: int,
    height: int,
    camera_id: int | None,
    segmentation: bool,
    markers,
) -> None:
    import mujoco

    viewport = mujoco.MjrRect(0, 0, width, height)
    if width > render_context.con.offWidth or height > render_context.con.offHeight:
        new_width = max(width, render_context.model.vis.global_.offwidth)
        new_height = max(height, render_context.model.vis.global_.offheight)
        render_context.update_offscreen_size(new_width, new_height)

    if camera_id is not None:
        render_context.cam.type = (
            mujoco.mjtCamera.mjCAMERA_FREE
            if camera_id == -1
            else mujoco.mjtCamera.mjCAMERA_FIXED
        )
        render_context.cam.fixedcamid = camera_id

    mujoco.mjv_updateScene(
        render_context.model._model,
        render_context.data._data,
        render_context.vopt,
        render_context.pert,
        render_context.cam,
        mujoco.mjtCatBit.mjCAT_ALL,
        render_context.scn,
    )
    if segmentation:
        render_context.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
        render_context.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

    _add_markers_to_scene(render_context, markers)
    mujoco.mjr_render(
        viewport=viewport,
        scn=render_context.scn,
        con=render_context.con,
    )

    if segmentation:
        render_context.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
        render_context.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0


def render_trajectory_frame(
    libero_env,
    reference: np.ndarray | None,
    executed: np.ndarray,
    marker_stride: int = 2,
) -> np.ndarray:
    """Render reference and inference trajectories as markers in the MuJoCo scene."""
    if reference is None or len(reference) == 0:
        return libero_env.render()

    markers = _trajectory_markers(np.asarray(reference), np.asarray(executed), marker_stride)
    sim = libero_env._env.sim
    render_context = sim._render_context_offscreen
    original_render = render_context.render

    def render(self, width, height, camera_id=None, segmentation=False):
        _render_with_markers(
            self,
            width,
            height,
            camera_id,
            segmentation,
            markers,
        )

    render_context.render = types.MethodType(render, render_context)
    try:
        frame = sim.render(
            camera_name="agentview",
            height=libero_env.observation_height,
            width=libero_env.observation_width,
        )
    finally:
        render_context.render = original_render
    return frame[::-1, ::-1]
