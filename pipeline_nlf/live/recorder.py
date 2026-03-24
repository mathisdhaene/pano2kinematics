import cv2


class YOLOTrackedRecorder:
    def __init__(self, model, out_path, fps=30.0, device="cuda:0", tracker="botsort.yaml", kp_conf_thresh=None):
        self.model = model
        self.out_path = str(out_path)
        self.fps = float(fps)
        self.device = device
        self.tracker = tracker
        self.kp_conf_thresh = kp_conf_thresh
        self.writer = None
        self.size = None

    def _ensure_writer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if self.writer is None:
            self.size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w, h))

    def process(self, frame_bgr):
        self._ensure_writer(frame_bgr)
        results = self.model.track(
            frame_bgr,
            device=self.device,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )[0]
        annotated = results.plot()
        if annotated.shape[1] != self.size[0] or annotated.shape[0] != self.size[1]:
            annotated = cv2.resize(annotated, self.size, interpolation=cv2.INTER_LINEAR)
        self.writer.write(annotated)
        return results

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
