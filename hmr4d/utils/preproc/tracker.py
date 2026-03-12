from ultralytics import YOLO
from hmr4d import PROJ_ROOT

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from hmr4d.utils.seq_utils import (
    get_frame_id_list_from_mask,
    linear_interpolate_frame_ids,
    frame_id_to_mask,
    rearrange_by_mask,
)
from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.net_utils import moving_average_smooth


class Tracker:
    def __init__(self) -> None:
        # https://docs.ultralytics.com/modes/predict/
        self.yolo = YOLO(PROJ_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")

    def track(self, video_path):
        track_history = []
        cfg = {
            "device": "cuda",
            "conf": 0.5,  # default 0.25, wham 0.5
            "classes": 0,  # human
            "verbose": False,
            "stream": True,
        }
        results = self.yolo.track(video_path, **cfg)
        # frame-by-frame tracking
        track_history = []
        for result in tqdm(results, total=get_video_lwh(video_path)[0], desc="YoloV8 Tracking"):
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()  # (N)
                bbx_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                result_frame = [{"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i]} for i in range(len(track_ids))]
            else:
                result_frame = []
            track_history.append(result_frame)

        return track_history

    @staticmethod
    def sort_track_length(track_history, video_path):
        """This handles the track history from YOLO tracker."""
        id_to_frame_ids = defaultdict(list)
        id_to_bbx_xyxys = defaultdict(list)
        # parse to {det_id : [frame_id]}
        for frame_id, frame in enumerate(track_history):
            for det in frame:
                id_to_frame_ids[det["id"]].append(frame_id)
                id_to_bbx_xyxys[det["id"]].append(det["bbx_xyxy"])
        for k, v in id_to_bbx_xyxys.items():
            id_to_bbx_xyxys[k] = np.array(v)

        # Sort by length of each track (max to min)
        id_length = {k: len(v) for k, v in id_to_frame_ids.items()}
        id2length = dict(sorted(id_length.items(), key=lambda item: item[1], reverse=True))

        # Sort by area sum (max to min)
        id_area_sum = {}
        l, w, h = get_video_lwh(video_path)
        for k, v in id_to_bbx_xyxys.items():
            bbx_wh = v[:, 2:] - v[:, :2]
            id_area_sum[k] = (bbx_wh[:, 0] * bbx_wh[:, 1] / w / h).sum()
        id2area_sum = dict(sorted(id_area_sum.items(), key=lambda item: item[1], reverse=True))
        id_sorted = list(id2area_sum.keys())

        return id_to_frame_ids, id_to_bbx_xyxys, id_sorted

    def get_one_track(self, video_path):
        # track
        track_history = self.track(video_path)

        # parse track_history & use top1 track
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(track_history, video_path)
        track_id = id_sorted[0]
        frame_ids = torch.tensor(id_to_frame_ids[track_id])  # (N,)
        bbx_xyxys = torch.tensor(id_to_bbx_xyxys[track_id])  # (N, 4)

        # interpolate missing frames
        mask = frame_id_to_mask(frame_ids, get_video_lwh(video_path)[0])
        bbx_xyxy_one_track = rearrange_by_mask(bbx_xyxys, mask)  # (F, 4), missing filled with 0
        missing_frame_id_list = get_frame_id_list_from_mask(~mask)  # list of list
        bbx_xyxy_one_track = linear_interpolate_frame_ids(bbx_xyxy_one_track, missing_frame_id_list)
        assert (bbx_xyxy_one_track.sum(1) != 0).all()

        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)

        return bbx_xyxy_one_track

    @staticmethod
    def _interpolate_and_smooth(frame_ids_list, bbx_xyxys_np, total_frames):
        """Interpolate missing frames and smooth a single track's bboxes.

        Same processing as get_one_track but factored out for reuse.
        Returns (bbx_xyxy_track, detection_mask) where bbx_xyxy_track is (F, 4)
        and detection_mask is (F,) bool indicating real vs interpolated frames.
        """
        frame_ids = torch.tensor(frame_ids_list)
        bbx_xyxys = torch.tensor(bbx_xyxys_np)

        mask = frame_id_to_mask(frame_ids, total_frames)
        bbx_xyxy_track = rearrange_by_mask(bbx_xyxys, mask)
        missing_frame_id_list = get_frame_id_list_from_mask(~mask)
        bbx_xyxy_track = linear_interpolate_frame_ids(bbx_xyxy_track, missing_frame_id_list)

        # For tracks that don't span the full video, some leading/trailing frames
        # may still be zero. Clamp them to the nearest valid bbox.
        nonzero = bbx_xyxy_track.sum(1) != 0
        if nonzero.any() and not nonzero.all():
            first_valid = nonzero.nonzero(as_tuple=True)[0][0].item()
            last_valid = nonzero.nonzero(as_tuple=True)[0][-1].item()
            if first_valid > 0:
                bbx_xyxy_track[:first_valid] = bbx_xyxy_track[first_valid]
            if last_valid < total_frames - 1:
                bbx_xyxy_track[last_valid + 1:] = bbx_xyxy_track[last_valid]

        bbx_xyxy_track = moving_average_smooth(bbx_xyxy_track, window_size=5, dim=0)
        bbx_xyxy_track = moving_average_smooth(bbx_xyxy_track, window_size=5, dim=0)

        return bbx_xyxy_track, mask

    def get_all_tracks(self, video_path, min_track_frames=30):
        """Track all people and return interpolated/smoothed bboxes for each.

        Args:
            video_path: Path to input video.
            min_track_frames: Minimum number of detected frames to keep a track.

        Returns:
            list of dicts, sorted by area (largest first). Each dict has:
                - track_id: int
                - bbx_xyxy: Tensor (F, 4) — interpolated & smoothed, full video length
                - detection_mask: Tensor (F,) bool — True for frames with real detections
        """
        track_history = self.track(video_path)
        total_frames = get_video_lwh(video_path)[0]

        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(
            track_history, video_path
        )

        results = []
        for track_id in id_sorted:
            if len(id_to_frame_ids[track_id]) < min_track_frames:
                continue

            bbx_xyxy_track, det_mask = self._interpolate_and_smooth(
                id_to_frame_ids[track_id],
                id_to_bbx_xyxys[track_id],
                total_frames,
            )

            results.append({
                "track_id": track_id,
                "bbx_xyxy": bbx_xyxy_track.float(),
                "detection_mask": det_mask,
            })

        return results
