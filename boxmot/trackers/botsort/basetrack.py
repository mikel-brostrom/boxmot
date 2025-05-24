from collections import OrderedDict

import numpy as np


class TrackState:
    """
    Enum-like class for tracking states.

    Attributes:
        New (int): Represents a newly created track.
        Tracked (int): Represents a currently tracked object.
        Lost (int): Represents a temporarily lost track.
        LongLost (int): Represents a track that has been lost for a long time.
        Removed (int): Represents a track that has been removed.
    """
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4


class BaseTrack:
    """
    Base class for managing the state of a track in multi-object tracking.

    Attributes:
        _count (int): Class variable to keep track of the number of tracks created.
        track_id (int): The unique ID assigned to the track.
        is_activated (bool): Whether the track has been activated.
        state (TrackState): The current state of the track.
        history (OrderedDict): A history of the track's past states or observations.
        features (list): A list of feature vectors associated with the track.
        curr_feature (np.ndarray): The most recent feature vector.
        score (float): The confidence score of the track.
        start_frame (int): The frame where the track started.
        frame_id (int): The most recent frame ID associated with the track.
        time_since_update (int): The number of frames since the track was last updated.
        location (tuple): The location of the object in multi-camera tracking (set to infinity by default).
    """
    _count = 0

    track_id: int = 0
    is_activated: bool = False
    state: int = TrackState.New

    history: OrderedDict = OrderedDict()
    features: list = []
    curr_feature: np.ndarray = None
    score: float = 0
    start_frame: int = 0
    frame_id: int = 0
    time_since_update: int = 0

    # multi-camera
    location: tuple = (np.inf, np.inf)

    @property
    def end_frame(self) -> int:
        """
        Returns the last frame the track was updated.

        Returns:
            int: The frame ID of the last update.
        """
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        """
        Generates the next unique track ID.

        Returns:
            int: A unique track ID.
        """
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """
        Activates the track. This method should be implemented in subclasses.

        Args:
            *args: Variable length argument list.

        Raises:
            NotImplementedError: If this method is not implemented in the subclass.
        """
        raise NotImplementedError

    def predict(self):
        """
        Predicts the next state of the track using a motion model. This method should be implemented in subclasses.

        Raises:
            NotImplementedError: If this method is not implemented in the subclass.
        """
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """
        Updates the state of the track based on a new observation. This method should be implemented in subclasses.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: If this method is not implemented in the subclass.
        """
        raise NotImplementedError

    def mark_lost(self):
        """
        Marks the track as lost.
        """
        self.state = TrackState.Lost

    def mark_long_lost(self):
        """
        Marks the track as long lost.
        """
        self.state = TrackState.LongLost

    def mark_removed(self):
        """
        Marks the track as removed.
        """
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        """
        Resets the track ID counter to 0.
        """
        BaseTrack._count = 0
