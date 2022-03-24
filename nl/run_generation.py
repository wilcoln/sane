
"""
Fine-tuning the library models for sequence to sequence.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import json

import nltk  
import numpy as np
from datasets import load_dataset, load_metric
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from wd_generation_utils import WeightedDecodingGenerationMixin
from tqdm import tqdm
def extend_instance(obj, cls):
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (base_cls, cls),{})


check_min_version("4.4.0")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    

    model_name_or_path: str = field(
        metadata={"help": "."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "."}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": ". "},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "."
        },
    )


@dataclass
class DataTrainingArguments:
    

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    
}


def main():


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "`--source_prefix 'summarize: ' `"
        )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)


    if data_args.dataset_name is not None:
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""


    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warn(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)


        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        # diff=[]
        # for i,_src in enumerate(model_inputs["input_ids"]):
        #     diff.append(list(set(_src).difference(set(labels["input_ids"][i]))))
        # model_inputs["diff"] = diff
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    
    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    #Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    def get_bag_of_words(inputs):
        bow=[]
        src=inputs["input_ids"].cpu().numpy()
        trg=inputs["labels"].cpu().numpy()
        for i,_src in enumerate(src):
            diff=set(trg[i]).difference(set(_src))
            if -100 in diff:
                diff.remove(-100)
            bow.append(list(diff))
        return bow
    def get_bag_of_all_words(inputs, now_set):
        now_set=set(now_set)
        src=inputs["input_ids"].cpu().numpy()
        trg=inputs["labels"].cpu().numpy()
        for i,_src in enumerate(src):
            diff=set(trg[i]).difference(set(_src))
            if -100 in diff:
                diff.remove(-100)
            now_set=now_set.union(diff)
        return list(now_set)

    if training_args.do_predict:
        logger.info("*** generate ***")
        extend_instance(model, WeightedDecodingGenerationMixin)

        gen_kwargs = {
            "max_length": model.config.max_length,
            "num_beams": model.config.num_beams,
        }
        test_results=[]
        test_dataloader=trainer.get_test_dataloader(test_dataset)
        bag_of_all_words=[]

        #!''' for all words
        use_wd=False
        all_words=False
        
        if all_words:
            print("preparing weighted decoding generation with all bag of words")
            for step, inputs in enumerate(tqdm(test_dataloader)):
                bag_of_all_words=get_bag_of_all_words(inputs, bag_of_all_words)

            # with open('dev_bad_words.json', 'r') as fp:
            #         jsonlist=fp.read()
            # bad_words_bag=json.loads(jsonlist)
            # bad_words_ids = [tokenizer(bad_word).input_ids for bags in bad_words_bag for bad_word in bags]
            logger.info("finish building bad words...")
        #!'''
        logger.info("=======start baseline generation...")
        for step, inputs in enumerate(tqdm(test_dataloader)):
            if use_wd is False:
                print("generation with BART generation")
                generated_tokens = model.generate(
                                inputs["input_ids"].cuda(),
                                attention_mask=inputs["attention_mask"].cuda(),
                                **gen_kwargs,
                            )
            else:
                if all_words is False:
                    print("generate with weighted decoding generation with each bag of words")
                    bag_of_words=get_bag_of_words(inputs)
                    generated_tokens = model.weighted_decoding_generate(
                        inputs["input_ids"].cuda(),
                        alpha=0.1,
                        beta=0.1,
                        bag_of_words_ids=bag_of_words,
                        do_sample=True,
                        top_k=10,
                        num_beams=1,
                        num_beam_groups=1,
                        bag_of_all_words_ids=None,
                        attention_mask=inputs["attention_mask"].cuda(),
                        # return_dict_in_generate = True,
                        #**gen_kwargs,
                    )
                else:
                    print("generate with weighted decoding generation with all bag of words")
                    generated_tokens = model.weighted_decoding_generate(
                        inputs["input_ids"].cuda(),
                        alpha=0.2,#0.1,
                        beta=0.2, #0.1,
                        bag_of_words_ids=None,
                        bag_of_all_words_ids=  bag_of_all_words,
                        # bad_words_ids = bad_words_ids, 
                        attention_mask=inputs["attention_mask"].cuda(),
                        **gen_kwargs,
                    )
            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    generated_tokens
                )
                test_results += [pred.replace('\n',' ').strip() for pred in test_preds]

        output_test_preds_file = os.path.join(training_args.output_dir, "test_generations.txt")
        with open(output_test_preds_file, "w") as writer:
            writer.write("\n".join(test_results))

        #!'''
        test_results=[]
        use_wd=True
        all_words=False
        logger.info("=======start weighted decoding generation...")
        for step, inputs in enumerate(tqdm(test_dataloader)):
            if use_wd is False:
                print("generation with BART generation")
                generated_tokens = model.generate(
                                inputs["input_ids"].cuda(),
                                attention_mask=inputs["attention_mask"].cuda(),
                                **gen_kwargs,
                            )
            else:
                if all_words is False:
                    #print("generate with weighted decoding generation with each bag of words")
                    bag_of_words=get_bag_of_words(inputs)
                    generated_tokens = model.weighted_decoding_generate(
                        inputs["input_ids"].cuda(),
                        alpha=0.1,
                        beta=0.1,
                        bag_of_words_ids=bag_of_words,
                        do_sample=True,
                        top_k=10,
                        num_beams=1,
                        num_beam_groups=1,
                        bag_of_all_words_ids=None,
                        attention_mask=inputs["attention_mask"].cuda(),
                        #**gen_kwargs,
                    )
                else:
                    print("generate with weighted decoding generation with all bag of words")
                    generated_tokens = model.weighted_decoding_generate(
                        inputs["input_ids"].cuda(),
                        alpha=0.1,#0.1,
                        beta=0.2, #0.1,
                        bag_of_words_ids=None,
                        bag_of_all_words_ids=  bag_of_all_words,
                        # bad_words_ids = bad_words_ids, 
                        attention_mask=inputs["attention_mask"].cuda(),
                        **gen_kwargs,
                    )
            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    generated_tokens
                )
                test_results += [pred.strip().replace('\n',' ') for pred in test_preds]

        output_test_preds_file = os.path.join(training_args.output_dir, "test_generations_wd.txt")
        with open(output_test_preds_file, "w") as writer:
            writer.write("\n".join(test_results))

    return 


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
