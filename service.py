import typing as t
from typing import Any, List

from PIL.Image import Image

import bentoml


BENTOML_MODEL_TAG = "resnet-50:kkhahjwedkfpqytn"

@bentoml.service(
    name="bentoresnet",
    traffic={
        "timeout": 300,
        "concurrency": 256,
    },
    # gpu의 경우 사용
    # resources={
    #     "gpu": 1,
    # },
)

class Resnet:
    bento_model_ref = bentoml.models.get(BENTOML_MODEL_TAG)

    def __init__(self) -> None:
        from transformers import AutoImageProcessor, ResNetForImageClassification
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ResNetForImageClassification.from_pretrained(
            self.bento_model_ref.path_of("model")
        ).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(
            self.bento_model_ref.path_of("processor")
        )
        print("Model resnet loaded", "device:", self.device)

    @bentoml.api(batchable=True, batch_dim=16)
    async def classify(self, images: List[Image]) -> dict[str, float | Any]:
        '''
        Classify input images to labels
        '''
        import torch

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits: torch.Tensor = self.model(**inputs).logits
        print(logits.shape)
        label_id = logits.squeeze(0).argmax().item()
        return {"label": self.model.config.id2label[label_id], "score": float(logits.max().item())}
