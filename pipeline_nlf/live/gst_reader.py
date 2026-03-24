import math
import os
import time

import numpy as np


def _pick_shm_paths(base: str):
    yield base
    for i in range(10):
        yield f"{base}.{i}"


class GstShmReader:
    """Minimal, robust shmsrc -> appsink BGR reader."""

    def __init__(self, sock_base, w, h, fps, timeout_ms=3000):
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as exc:
            raise RuntimeError(
                "GStreamer bindings (gi) are required for --live. "
                "Install python3-gi / pygobject or run without --live."
            ) from exc

        self.Gst = Gst
        Gst.init(None)
        self.w, self.h = int(w), int(h)
        self.pipeline = None
        self.sink = None
        self.bus = None
        self._closed = False
        self._logged_size = False
        self.try_pull_timeout_ns = 500_000_000

        caps = "video/x-raw,format=BGR"
        last_err = None

        for path in _pick_shm_paths(sock_base):
            deadline = time.time() + 3.0
            while not os.path.exists(path) and time.time() < deadline:
                time.sleep(0.05)
            if not os.path.exists(path):
                continue

            pipeline = None
            pipeline_description = (
                f"shmsrc socket-path={path} is-live=true do-timestamp=true ! "
                f"{caps} ! "
                "appsink name=sink drop=true sync=false max-buffers=1 emit-signals=true"
            )
            try:
                pipeline = Gst.parse_launch(pipeline_description)
                sink = pipeline.get_by_name("sink")
                bus = pipeline.get_bus()
                ret = pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    msg = bus.timed_pop_filtered(timeout_ms * Gst.MSECOND, Gst.MessageType.ERROR)
                    if msg:
                        last_err = msg.parse_error()
                        pipeline.set_state(Gst.State.NULL)
                        continue
                    pipeline.set_state(Gst.State.NULL)
                    continue
                self.pipeline, self.sink, self.bus = pipeline, sink, bus
                print(f"[GstShmReader] Connected to {path}")
                break
            except Exception as exc:
                last_err = exc
                try:
                    pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass

        if self.pipeline is None:
            if isinstance(last_err, tuple):
                err, dbg = last_err
                raise RuntimeError(f"Failed to open shmsrc: {err.message} [{dbg}]")
            raise RuntimeError(f"Failed to open any shmsrc under {sock_base}")

    def read(self):
        Gst = self.Gst
        if self._closed or self.pipeline is None:
            return False, None
        if self.bus:
            msg = self.bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg:
                return False, None

        sample = self.sink.emit("try-pull-sample", self.try_pull_timeout_ns)
        if not sample:
            return False, None

        width = height = None
        caps = sample.get_caps()
        if caps:
            structure = caps.get_structure(0)
            if structure:
                width = structure.get_value("width")
                height = structure.get_value("height")

        buffer = sample.get_buffer()
        ok, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return False, None

        arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
        pixels = arr.size // 3
        if width is None or height is None:
            guessed_height = int(round(math.sqrt(max(pixels, 0) / 2)))
            guessed_width = 2 * guessed_height
            if guessed_width * guessed_height == pixels and guessed_height > 0:
                width, height = guessed_width, guessed_height
            else:
                width, height = self.w, self.h

        expected = int(width) * int(height) * 3
        if arr.size != expected:
            buffer.unmap(mapinfo)
            return False, None

        frame = arr.reshape((int(height), int(width), 3)).copy()
        buffer.unmap(mapinfo)
        if not self._logged_size:
            print(f"[GstShmReader] negotiated size: {width}x{height}")
            self._logged_size = True
        return True, frame

    def close(self):
        Gst = self.Gst
        if self._closed:
            return
        self._closed = True
        try:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.PAUSED)
                self.pipeline.set_state(Gst.State.READY)
                self.pipeline.set_state(Gst.State.NULL)
        finally:
            self.pipeline = None
            self.sink = None
            try:
                Gst.deinit()
            except Exception:
                pass
