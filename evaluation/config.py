from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ModelConfig:
    base: Optional[str] = None
    path: str = ""
    name: str = ""


@dataclass
class DataConfig:
    directory: str = ""
    file_path: str = ""
    question_file: str = ""


@dataclass
class ResultsConfig:
    directory: str = ""
    unique_id: str = "default"


@dataclass
class EvalParams:
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    results: ResultsConfig = ResultsConfig()
    strategy: str = ""


@dataclass
class LlavaConfig:
    conv_mode: str = ""
    temperature: float = 0.0
    top_p: Optional[float] = None
    num_beams: int = 1


@dataclass
class DatasetConfig:
    name: str
    # params: Dict[str, Any] = field(default_factory=dict)
    params: EvalParams = EvalParams()


@dataclass
class GradeConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    dataset: DatasetConfig
    grader: GradeConfig
    llava_config: LlavaConfig
