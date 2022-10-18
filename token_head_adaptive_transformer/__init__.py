# Length-Adaptive Transformer
# Copyright (c) 2020-present NAVER Corp.
# Apache License v2.0

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    BertConfig,
    DistilBertConfig,
    MobileBertConfig,
)

from .modeling_bert import (
    BertForSequenceClassification,
    BertForQuestionAnswering,
)
from .modeling_distilbert import (
    DistilBertForSequenceClassification,
    DistilBertForQuestionAnswering,
)
from .modeling_mobilebert import (
    MobileBertForSequenceClassification,
    MobileBertForQuestionAnswering,
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.update(
    [
        (BertConfig, BertForSequenceClassification),
        (DistilBertConfig, DistilBertForSequenceClassification),
        (MobileBertConfig, MobileBertForSequenceClassification),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING.update(
    [
        (BertConfig, BertForQuestionAnswering),
        (DistilBertConfig, DistilBertForQuestionAnswering),
        (MobileBertConfig, MobileBertForQuestionAnswering),
    ]
)

from .training_args import TrainingArguments
from .drop_and_restore_utils import LengthDropArguments, SearchArguments
from .trainer import LengthDropTrainer
