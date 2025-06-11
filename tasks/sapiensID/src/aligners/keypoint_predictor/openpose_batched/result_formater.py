from typing import List, Tuple
import numpy as np
from .open_pose import PoseResult, BodyResult, Keypoint


def format_body_result(candidate: np.ndarray, subset: np.ndarray) -> List[BodyResult]:
    return [
        BodyResult(
            keypoints=[
                Keypoint(
                    x=candidate[candidate_index][0],
                    y=candidate[candidate_index][1],
                    score=candidate[candidate_index][2],
                    id=candidate[candidate_index][3]
                ) if candidate_index != -1 else None
                for candidate_index in person[:18].astype(int)
            ],
            total_score=person[18],
            total_parts=person[19]
        )
        for person in subset
    ]

def make_pose_result(bodies: List[BodyResult], orig_shape: Tuple[int, int], include_hand: bool=False, include_face: bool=False):
    results = []
    H, W = orig_shape
    for body in bodies:
        left_hand, right_hand, face = (None,) * 3
        if include_hand:
            # left_hand, right_hand = self.detect_hands(body, oriImg)
            raise NotImplementedError
        if include_face:
            # face = self.detect_face(body, oriImg)
            raise NotImplementedError

        results.append(PoseResult(BodyResult(
            keypoints=[
                Keypoint(
                    x=keypoint.x / float(W),
                    y=keypoint.y / float(H)
                ) if keypoint is not None else None
                for keypoint in body.keypoints
            ], 
            total_score=body.total_score,
            total_parts=body.total_parts
        ), left_hand, right_hand, face))
    return results


def select_best_pose(poses: List[PoseResult]):
    # order according to body_score
    if len(poses) > 0:
        pose_result = sorted(poses, key=lambda x: x.body.total_score, reverse=True)[0]  # descending
    else:
        # make null pose
        pose_result = PoseResult(body=BodyResult(keypoints=[None]*18, total_score=0, total_parts=0), face=None, left_hand=None, right_hand=None)
    return pose_result