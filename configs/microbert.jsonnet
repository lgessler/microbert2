// --------------------------------------------------------------------------------
// Parameters
// --------------------------------------------------------------------------------
local language = "wolof";
// Optional, will be used in generated files--set to something descriptive if you'd like.
local experiment_name = "44";

// Tokenization -------------------------------------------------------------------
// Do you want Stanza to retokenize your input? Set to `false` if you are confident
// in the quality of your tokenization, or if your language is not supported by Stanza.
local stanza_retokenize = true;
// Do you want Stanza to look for multi-word tokens? See https://stanfordnlp.github.io/stanza/mwt.html
// You probably want to keep this at `false`, since in many languges, the subtokens
// can differ quite a bit from surface forms (e.g. aux => à + les in French)
local stanza_use_mwt = false;
// Only needed if stanza_retokenize is `true`. Find your language code here:
// https://stanfordnlp.github.io/stanza/performance.html
local stanza_language_code = "wo";
// If set to null, we will attempt to guess something sensible. For reference, BERT
// has 30000 vocabulary items.
local vocab_size = 10000;

// Data ---------------------------------------------------------------------------
local whitespace_tokenized_text_path_train = "data/wolof/converted_punct/train/train.txt";
local whitespace_tokenized_text_path_dev = "data/wolof/converted_punct/dev/dev.txt";

// Encoder ------------------------------------------------------------------------
local max_length = 512;
local hidden_size = 128;
local num_layers = 3;
// See https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertConfig
// Note that vocab_size will be overridden, so do not set it based on your tokenizer settings,
// so do not overwrite it.
local bert_config = {
    hidden_size: hidden_size,
    num_hidden_layers: num_layers,
    num_attention_heads: 8,
    intermediate_size: 512,
    max_position_embeddings: max_length,
};

// Training -----------------------------------------------------------------------
local batch_size = 64;
local grad_accum = 8;
local effective_batch_size = grad_accum * batch_size;
local num_steps = 1e4;
local validate_every = 1000;  // in steps

// Optimizer ----------------------------------------------------------------------
local optimizer = {
    type: "torch::AdamW",
    lr: 3e-4,
    betas: [0.9, 0.98],
    eps: 1e-6,
    weight_decay: 0.01
};
local lr_scheduler = {
    type: "transformers::cosine",
    num_warmup_steps: validate_every,
    num_training_steps: num_steps,
};

// Tasks --------------------------------------------------------------------------


// --------------------------------------------------------------------------------
// Internal--don't modify below here unless you're sure you know what you're doing!
// --------------------------------------------------------------------------------
local util = import 'lib/util.libsonnet';
local model_path = (
    "./workspace/models/" + language + "_" + experiment_name + "_"+ util.stringifyObject(bert_config)
);
local tokenizer = { pretrained_model_name_or_path: model_path };
local model = {
    type: "microbert2.microbert.model.model::microbert_model",
    tokenizer: tokenizer,
    model_output_path: model_path,
    encoder: {
        type: "bert",
        tokenizer: tokenizer,
        bert_config: bert_config,
    }
};

local training_engine = {
    type: "torch",
    optimizer: optimizer,
    lr_scheduler: lr_scheduler,
    amp: false
};

local collate_fn = {
    type: "microbert2.microbert.collator::collator",
    tokenizer: tokenizer,
    // whether to replace [MASK] with 10% UNK and 10% random. should be true for electra, false for bert
    mask_only: false,
};
local train_dataloader = {
    shuffle: true,
    batch_size: batch_size,
    collate_fn: collate_fn,
    pin_memory: true,
    num_workers: 2,
    prefetch_factor: 4,
    persistent_workers: true,
};
local val_dataloader = {
    shuffle: false,
    batch_size: batch_size,
    collate_fn: collate_fn,
    pin_memory: true,
    num_workers: 2,
    prefetch_factor: 4,
    persistent_workers: true,
};

{
    steps: {
        raw_text_data: {
            type: "microbert2.data.text::read_whitespace_tokenized_text",
            train_path: whitespace_tokenized_text_path_train,
            dev_path: whitespace_tokenized_text_path_dev,
            stanza_retokenize: stanza_retokenize,
            stanza_use_mwt: stanza_use_mwt,
            stanza_language_code: stanza_language_code,
        },
        tokenizer: {
            type: "microbert2.data.tokenize::train_tokenizer",
            dataset: { "type": "ref", "ref": "raw_text_data" },
            vocab_size: vocab_size,
            model_path: model_path
        },
        tokenized_text_data: {
            type: "microbert2.data.tokenize::subword_tokenize",
            dataset: { "type": "ref", "ref": "raw_text_data" },
            max_length: max_length,
            tokenizer: tokenizer,
            step_extra_dependencies: [ {type: "ref", "ref": "tokenizer" } ]
        },

        // Merge inputs
        model_inputs: {
            type: "microbert2.microbert.data::combine_datasets",
            dataset: { "type": "ref", "ref": "tokenized_text_data" },
        },
        // // Record label counts
        // counts: {
        //     type: "microbert2.data.util::count_unique_values",
        //     dataset: { "type": "ref", "ref": "model_inputs" },
        //     keys: ["xpos", "deprel"],
        // },

        // Begin training
        trained_model: {
            type: "microbert2.train::train",
            model: model,
            dataset_dict: { type: "ref", ref: "model_inputs" },
            training_engine: training_engine,
            log_every: 1,
            train_dataloader: train_dataloader,
            //train_epochs: num_epochs,
            train_steps: num_steps,
            grad_accum: grad_accum,
            validate_every: validate_every,
            checkpoint_every: validate_every,
            validation_split: "dev",
            validation_dataloader: val_dataloader,
            // val_metric_name: "perplexity",
            // minimize_val_metric: true,
            callbacks: [
                {
                    type: "microbert2.microbert.model.model::write_model",
                    path: model_path,
                    model_attr: "encoder.encoder"
                }
            ],
        },
        //final_metrics: {
        //    type: "torch::eval",
        //    model: { type: "ref", ref: "trained_model" },
        //    dataset_dict: { type: "ref", ref: "stype_instances" },
        //    dataloader: val_dataloader,
        //    metric_names: ["loss", "accuracy"],
        //    test_split: "test",
        //},
    }
}
