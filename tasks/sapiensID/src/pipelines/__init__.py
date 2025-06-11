from typing import Any
from .infer_aligner_model_pipeline import InferAlignerModelPipeline


def pipeline_from_name(name: str, model: Any=None, aligner: Any=None, 
                       single_image_pipeline=None, aggregator=None,
                       extra_image_pipeline=None,):
    if name == 'infer_aligner_model_pipeline':
        pipeline = InferAlignerModelPipeline(model=model, aligner=aligner)
    else:
        raise NotImplementedError(f"pipeline {name} not implemented")

    return pipeline