from __future__ import print_function

import copy
from enum import Enum
import time


class MissionState(Enum):
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    PLANNING = "PLANNING"
    VERIFY_TARGET = "VERIFY_TARGET"
    SEARCHING = "SEARCHING"
    MAP_NAVIGATING = "MAP_NAVIGATING"
    ALIGNING = "ALIGNING"
    APPROACHING = "APPROACHING"
    AVOIDING = "AVOIDING"
    FINAL_VERIFY = "FINAL_VERIFY"
    BIN_SIDE_DOCKING = "BIN_SIDE_DOCKING"
    FINALIZING = "FINALIZING"
    INTERMEDIATE = "INTERMEDIATE"
    DONE = "DONE"
    FAILED = "FAILED"


class MissionEvent(Enum):
    START = "START"
    INITIALIZED = "INITIALIZED"
    TARGET_REQUIRED = "TARGET_REQUIRED"
    MAP_TARGET_AVAILABLE = "MAP_TARGET_AVAILABLE"
    TARGET_FOUND = "TARGET_FOUND"
    TARGET_MISSING = "TARGET_MISSING"
    TARGET_ALIGNED = "TARGET_ALIGNED"
    TARGET_REACHED = "TARGET_REACHED"
    TARGET_STABLE = "TARGET_STABLE"
    SIDE_DOCK_REQUIRED = "SIDE_DOCK_REQUIRED"
    MAP_DESTINATION_REACHED = "MAP_DESTINATION_REACHED"
    REPLAN = "REPLAN"
    OBSTACLE_FOUND = "OBSTACLE_FOUND"
    PATH_CLEAR = "PATH_CLEAR"
    ROUTINE_RESUME = "ROUTINE_RESUME"
    FINALIZED = "FINALIZED"
    RETRY = "RETRY"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    TIMEOUT = "TIMEOUT"
    FAIL = "FAIL"


class TargetType(Enum):
    CAN = "can"
    BIN = "bin"


class MissionContext(object):
    """Mutable mission memory shared by state handlers."""

    def __init__(self):
        self.vague_map = None
        self.distance_to_target = None
        self.grabbed = False
        self.completed_pickups = 0
        self.target_found = False
        self.obstacle_found = False
        self.target_type = None
        self.last_observation = None
        self.last_error = None
        self.retry_counts = {}
        self.state_data = {}
        self.searching_routine_data = {}
        self.history = []
        self.metrics = {
            "state_seconds": {},
            "detector_calls": {},
            "detector_seconds": {},
        }
        self._pickup_counted = False

    def begin_state(self, state):
        self.state_data = {
            "entered_at": time.time(),
            "steps": 0,
            "lost_frames": 0,
        }
        self.history.append({"state": state.value, "timestamp": time.time()})

    def finish_state(self, event):
        if not self.history:
            return
        entry = self.history[-1]
        elapsed = self.elapsed()
        entry["event"] = event.value
        entry["elapsed_seconds"] = elapsed
        entry["steps"] = int(self.state_data.get("steps", 0))
        name = entry["state"]
        self.metrics["state_seconds"][name] = float(self.metrics["state_seconds"].get(name, 0.0)) + elapsed

    def record_detection(self, target_type, elapsed_seconds):
        name = target_type.value
        self.metrics["detector_calls"][name] = int(self.metrics["detector_calls"].get(name, 0)) + 1
        self.metrics["detector_seconds"][name] = (
            float(self.metrics["detector_seconds"].get(name, 0.0)) + float(elapsed_seconds)
        )

    def increment_step(self):
        self.state_data["steps"] = int(self.state_data.get("steps", 0)) + 1
        return self.state_data["steps"]

    def elapsed(self):
        return time.time() - float(self.state_data.get("entered_at", time.time()))

    def remember_target(self, target_type, observation):
        self.target_type = target_type
        self.target_found = True
        self.last_observation = observation

    def forget_target(self, target_type=None):
        target_type = target_type or self.target_type
        self.target_found = False
        self.last_observation = None
        self.distance_to_target = None

    def mark_pickup(self):
        self.grabbed = True
        if not self._pickup_counted:
            self.completed_pickups += 1
            self._pickup_counted = True

    def mark_release(self):
        self.grabbed = False
        self._pickup_counted = False

    def retry(self, state):
        key = state.value
        self.retry_counts[key] = int(self.retry_counts.get(key, 0)) + 1
        return self.retry_counts[key]

    def snapshot(self):
        return {
            "vague_map": self.vague_map.snapshot() if self.vague_map is not None else None,
            "distance_to_target": self.distance_to_target,
            "grabbed": self.grabbed,
            "completed_pickups": self.completed_pickups,
            "target_found": self.target_found,
            "obstacle_found": self.obstacle_found,
            "target_type": self.target_type.value if self.target_type else None,
            "last_error": self.last_error,
            "retry_counts": dict(self.retry_counts),
            "searching_routine": copy.deepcopy(self.searching_routine_data),
            "metrics": self.metrics,
        }
