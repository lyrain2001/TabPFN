from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import Self

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import softmax

from tabpfn.base import (
    create_inference_engine,
    determine_precision,
    initialize_tabpfn_model,
)
from tabpfn.config import ModelInterfaceConfig
from tabpfn.constants import (
    PROBABILITY_EPSILON_ROUND_ZERO,
    SKLEARN_16_DECIMAL_PRECISION,
    XType,
    YType,
)
from tabpfn.preprocessing import (
    ClassifierEnsembleConfig,
    EnsembleConfig,
    default_classifier_preprocessor_configs,
)
from tabpfn.utils import (
    _fix_dtypes,
    _get_embeddings,
    _get_ordinal_encoder,
    infer_categorical_features,
    infer_device_and_type,
    infer_random_state,
    update_encoder_outlier_params,
    validate_X_predict,
    validate_Xy_fit,
)

from tabpfn.inference import (
    InferenceEngine,
    InferenceEngineCachePreprocessing,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.compose import ColumnTransformer
    from torch.types import _dtype
    from tabpfn.inference import InferenceEngine
    from tabpfn.model.config import InferenceConfig


class TabPFNEncoder(ClassifierMixin, BaseEstimator):
    """
    Minimal TabPFNEncoder that can do:
      1) init_model(): loads a PerFeatureTransformer into self.model_
      2) get_raw_embeddings(X): extracts embeddings for each sample
      3) predict_in_context(X_train, y_train, X_test): single forward pass
         that treats X_train,y_train as "training block" and X_test as "test block."
    """

    def __init__(
        self,
        *,
        model_path: str | Path | Literal["auto"] = "auto",
        device: str | torch.device | Literal["auto"] = "auto",
        inference_precision: _dtype | Literal["autocast", "auto"] = "auto",
        fit_mode: Literal[
            "low_memory",
            "fit_preprocessors",
            "fit_with_cache",
        ] = "fit_preprocessors",
        random_state: int | np.random.RandomState | np.random.Generator | None = 0,
    ) -> None:

        super().__init__()
        self.model_path = model_path
        self.device = device
        self.inference_precision: torch.dtype | Literal["autocast", "auto"] = (
            inference_precision
        )
        self.fit_mode: Literal[
            "low_memory",
            "fit_preprocessors",
            "fit_with_cache",
        ] = fit_mode
        self.random_state = random_state

    def _more_tags(self) -> dict[str, Any]:
        return {
            "allow_nan": True,
            "multilabel": False,
        }

    def init_model(self) -> Self:
        """
        Load a pre-trained TabPFN model (PerFeatureTransformer) 
        but do NOT do any normal "fit" on user data. Just load.
        """
        static_seed, rng = infer_random_state(self.random_state)
        self.model_, self.config_, _ = initialize_tabpfn_model(
            model_path=self.model_path,
            which="classifier",
            fit_mode=self.fit_mode,
            static_seed=static_seed,
        )
        self.device_ = infer_device_and_type(self.device)
        (self.use_autocast_, self.forced_inference_dtype_, byte_size) = (
            determine_precision(self.inference_precision, self.device_)
        )
        return self

    def get_raw_embeddings(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Directly calls the underlying PerFeatureTransformer on X to get embeddings
        without a scikit-learn `.fit()`.
        """
        if not hasattr(self, "model_"):
            raise ValueError(
                "You must call `encoder.init_model()` first so that `encoder.model_` "
                "is a loaded PerFeatureTransformer."
            )

        model = self.model_
        device = self.device_

        model.to(device)
        model.eval()

        # Reshape X to (seq_len, batch_size=1, n_features)
        n_samples, n_features = X.shape
        X_torch = torch.tensor(X, dtype=torch.float32, device=device)
        X_torch = X_torch.reshape(n_samples, 1, n_features)

        # Dummy y of shape (seq_len, 1)
        y_torch = torch.zeros((n_samples, 1), dtype=torch.float32, device=device)

        # single_eval_pos = entire length => store embeddings in "train_embeddings"
        single_eval_pos = n_samples

        with torch.inference_mode():
            output_dict = model(
                X_torch,
                y_torch,
                single_eval_pos=single_eval_pos,
                only_return_standard_out=False,  # retrieve embeddings
            )

        # "train_embeddings" has shape (seq_len, 1, embedding_dim)
        train_embeddings = output_dict["train_embeddings"]  # a torch.Tensor
        train_embeddings = train_embeddings.squeeze(1)      # => (seq_len, embedding_dim)

        return train_embeddings.cpu().numpy()

    def predict_in_context(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Use the raw PerFeatureTransformer to do classification in a single forward pass:
          - The first block (X_train,y_train) is the "training" context
          - The second block (X_test) is the "test" query
        Returns predicted class labels for X_test.

        This bypasses typical scikit-learn fit/predict. 
        We never store or remember X_train,y_train in `self` — everything is done
        in a single shot, in-context.
        """
        if not hasattr(self, "model_"):
            raise ValueError(
                "You must call `encoder.init_model()` first so that `encoder.model_` "
                "is a loaded PerFeatureTransformer."
            )

        model = self.model_
        device = self.device_
        model.to(device)
        model.eval()

        n_train = X_train.shape[0]
        n_test  = X_test.shape[0]

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device).unsqueeze(1)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
        X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device).unsqueeze(1)

        y_test_placeholder = torch.zeros((n_test, 1), dtype=torch.float32, device=device)

        # Concatenate train & test along the "sequence" dimension (dim=0)
        X_full = torch.cat([X_train_t, X_test_t], dim=0)           # (n_train + n_test, 1, n_features)
        y_full = torch.cat([y_train_t, y_test_placeholder], dim=0) # (n_train + n_test, 1)

        # single_eval_pos => treat first block as "train," second block as "test"
        single_eval_pos = n_train

        with torch.inference_mode():
            # "only_return_standard_out=True" => model returns just the "predictions"
            # (the standard decoder output). Typically shape: (n_test, 1, n_classes).
            logits = model(
                X_full,
                y_full,
                single_eval_pos=single_eval_pos,
                only_return_standard_out=True,
            )
            # -> shape: (n_test, 1, n_classes) because only the last n_test are predictions

        #    If the model is binary with 2 outputs => shape (n_test, 1, 2)
        #    For multi-class => shape (n_test, 1, num_classes)
        probs = softmax(logits, dim=-1)  # shape: (n_test, 1, num_classes)
        predictions = probs.argmax(dim=-1).squeeze(1).cpu().numpy()  # (n_test,)

        return predictions
