import asyncio
import queue
import threading
import time

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription


class WhepReader:
    """Background aiortc receiver that exposes cv2-style read()."""

    def __init__(self, whep_url, timeout_s=10.0, queue_size=2):
        self.whep_url = whep_url
        self.timeout_s = float(timeout_s)
        self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._thread = None
        self._error = None
        self._pc = None

    async def _receive_video(self, track):
        while not self._stop.is_set():
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put_nowait(img)

    async def _run(self):
        self._pc = RTCPeerConnection()
        self._pc.addTransceiver("video", direction="recvonly")

        @self._pc.on("track")
        def on_track(track):
            if track.kind == "video":
                asyncio.create_task(self._receive_video(track))

        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.whep_url,
                data=self._pc.localDescription.sdp,
                headers={"Content-Type": "application/sdp"},
            ) as resp:
                if resp.status != 201:
                    err = await resp.text()
                    raise RuntimeError(f"WHEP connect failed: {resp.status} - {err}")
                answer_sdp = await resp.text()

        answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
        await self._pc.setRemoteDescription(answer)
        self._connected.set()

        while not self._stop.is_set():
            await asyncio.sleep(0.05)

    def _thread_main(self):
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self._error = exc
        finally:
            if self._pc is not None:
                try:
                    asyncio.run(self._pc.close())
                except Exception:
                    pass

    def start(self):
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if self._error is not None:
                raise RuntimeError(f"WHEP reader failed: {self._error}") from self._error
            if self._connected.is_set():
                print(f"[WHEP] Connected to {self.whep_url}")
                return
            time.sleep(0.05)
        raise TimeoutError(f"WHEP connect timeout ({self.timeout_s:.1f}s): {self.whep_url}")

    def read(self, timeout_s=0.5):
        if self._error is not None:
            return False, None
        try:
            frame = self._queue.get(timeout=float(timeout_s))
            return True, frame
        except queue.Empty:
            return False, None

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def error(self):
        return self._error
