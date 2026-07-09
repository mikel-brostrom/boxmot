# BoxMOT AGPL-3.0 license

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseValidator(ABC):
    """A base class for creating validators.

    This class provides the foundation for validation processes, including
    model evaluation, metric computation, and result visualization.
    """

    def __init__(self, args: Any | None = None) -> None:
        self.args = args

    def __call__(self) -> Any:
        """Run the full validation process."""
        return self.validate()

    def validate(self) -> Any:
        """Execute setup, prediction, metric evaluation, and finalization."""
        self.setup()
        predictions = self.predict()
        results = self.evaluate(predictions)
        return self.finalize(results)

    @abstractmethod
    def setup(self) -> None:
        """Prepare model, data, and validation state."""

    @abstractmethod
    def predict(self) -> Any:
        """Run model inference needed for validation."""

    @abstractmethod
    def evaluate(self, predictions: Any) -> Any:
        """Compute metrics from predictions."""

    def finalize(self, results: Any) -> Any:
        """Persist, visualize, or otherwise process validation results."""
        return results


class BasePredictor(ABC):
    """A base class for creating predictors.

    This class provides the foundation for embedding generation functionality,
    handling model setup, inference, and result processing.
    """

    def __init__(self, args: Any | None = None) -> None:
        self.args = args
        self.model = None

    def __call__(self, *args, **kwargs) -> Any:
        """Run prediction."""
        return self.predict(*args, **kwargs)

    def setup_model(self) -> Any:
        """Build or return the model used for prediction."""
        return self.model

    def predict(self, *args, **kwargs) -> Any:
        """Run setup, inference, and postprocessing."""
        if self.model is None:
            self.model = self.setup_model()
        raw = self.inference(*args, **kwargs)
        return self.postprocess(raw)

    @abstractmethod
    def inference(self, *args, **kwargs) -> Any:
        """Run model inference."""

    def postprocess(self, predictions: Any) -> Any:
        """Process raw prediction outputs."""
        return predictions
