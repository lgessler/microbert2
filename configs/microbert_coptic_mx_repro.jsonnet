// --------------------------------------------------------------------------------
// Parameters
// --------------------------------------------------------------------------------
local language = "coptic";
// Optional and purely descriptive, intended to help you keep track of different model
// configurations. Set to `""` if you don't want to bother.
local experiment_name = "mx_repro";

// Tokenization -------------------------------------------------------------------
// Do you want Stanza to retokenize your input? Set to `false` if you are confident
// in the quality of your tokenization, or if your language is not supported by Stanza.
local stanza_retokenize = false;
// Do you want Stanza to look for multi-word tokens? See https://stanfordnlp.github.io/stanza/mwt.html
// You probably want to keep this at `false`, since in many languges, the subtokens
// can differ quite a bit from surface forms (e.g. aux => à + les in French)
local stanza_use_mwt = false;
// Only needed if stanza_retokenize is `true`. Find your language code here:
// https://stanfordnlp.github.io/stanza/performance.html
local stanza_language_code = null;
// If set to null, we will attempt to guess something sensible. For reference, BERT
// has 30000 vocabulary items.
local vocab_size = 10000;

// Data ---------------------------------------------------------------------------
local whitespace_tokenized_text_path_train = "data/cop/train.txt";
local whitespace_tokenized_text_path_dev = "data/cop/dev.txt";
local train_conllu_path = "data/cop/cop_scriptorium-ud-train.conllu";
local dev_conllu_path = "data/cop/cop_scriptorium-ud-dev.conllu";
local test_conllu_path = "data/cop/cop_scriptorium-ud-test.conllu";

// Encoder ------------------------------------------------------------------------
local max_length = 512;
local hidden_size = 100;
local num_layers = 3;
// Type of encoder stack. See microbert2/microbert/model/encoder.py for implementations.
local bert_type = "bert";
// local bert_type = "modernbert";
// local bert_type = "electra";

// Encoder stack configuration.
// See https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertConfig
// Note that vocab_size will be overridden, so do not set it based on your tokenizer settings,
// so do not overwrite it.
local bert_config = {
    hidden_size: hidden_size,
    num_hidden_layers: num_layers,
    num_attention_heads: 5,
    intermediate_size: std.floor(hidden_size * 2.5),
    max_position_embeddings: max_length,
};

// Training and Optimization ------------------------------------------------------
local batch_size = 32;
local grad_accum = 1;
local effective_batch_size = grad_accum * batch_size;
// local num_steps = 1e5;
local num_epochs = 200;

local optimizer = {
    type: "torch::AdamW",
    lr: 3e-3,
    betas: [0.9, 0.999],
    eps: 1e-6,
    weight_decay: 0.05
};

local lr_scheduler = {
    type: "transformers::cosine",
    num_warmup_steps: 1152 * 5,
    num_training_steps: 1152 * 200,
};

// Some set up, don't modify ------------------------------------------------------
local util = import 'lib/util.libsonnet';
local model_path = (
    "./workspace/models/" + language + "_" + experiment_name + "_" + util.stringifyObject(bert_config)
);
local tokenizer = { pretrained_model_name_or_path: model_path };

// Tasks --------------------------------------------------------------------------
// We provide an explicit abstraction for tasks which allows you to program them
// in a modular way. Each task needs a dataset, a head, and some other information.
// See microbert2/microbert/tasks/task.py.
local mlm_task = {
    type: "microbert2.microbert.tasks.mlm.MLMTask",
    dataset: { type: "ref", ref: "raw_text_data" },
    tokenizer: tokenizer,
};
local pos_task = {
    type: "microbert2.microbert.tasks.ud_pos.UDPOSTask",
    head: {
        num_layers: num_layers,
        embedding_dim: hidden_size,
        use_layer_mix: false,
        layer_index: 1,
    },
    tag_type: "xpos",
    train_conllu_path: train_conllu_path,
    dev_conllu_path: dev_conllu_path,
    test_conllu_path: test_conllu_path,
};
local parser = (import "lib/parser.libsonnet")(hidden_size, num_layers);
local parse_task = {
    type: "microbert2.microbert.tasks.ud_parse.UDParseTask",
    head: parser,
    train_conllu_path: train_conllu_path,
    dev_conllu_path: dev_conllu_path,
    test_conllu_path: test_conllu_path,
};
local tasks = [mlm_task, pos_task];


// --------------------------------------------------------------------------------
// Internal--don't modify below here unless you're sure you know what you're doing!
// --------------------------------------------------------------------------------
local model = {
    type: "microbert2.microbert.model.model::microbert_model",
    tokenizer: tokenizer,
    model_output_path: model_path,
    tasks: tasks,
    encoder: {
        type: bert_type,
        tokenizer: tokenizer,
        bert_config: bert_config,
    }
};

local training_engine = {
    type: "torch",
    optimizer: optimizer,
    lr_scheduler: lr_scheduler,
    amp: false,
    max_grad_norm: 1.0,
};

local collate_fn = {
    type: "microbert2.data.collator::collator",
    tokenizer: tokenizer,
    tasks: tasks,
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
            test_path: whitespace_tokenized_text_path_dev,
            stanza_retokenize: stanza_retokenize,
            stanza_use_mwt: stanza_use_mwt,
            stanza_language_code: stanza_language_code,
        },
        tokenizer: {
            type: "microbert2.data.tokenize::train_tokenizer",
            dataset: { "type": "ref", "ref": "raw_text_data" },
            vocab_size: vocab_size,
            model_path: model_path,
            tasks: tasks,
        },
        tokenized_text_data: {
            type: "microbert2.data.tokenize::subword_tokenize",
            dataset: { "type": "ref", "ref": "raw_text_data" },
            max_length: max_length,
            tokenizer: { "type": "ref", "ref": "tokenizer" },
            tasks: tasks,
        },

        // Merge inputs
        model_inputs: {
            type: "microbert2.data.combine::combine_datasets",
            datasets: { "type": "ref", "ref": "tokenized_text_data" },
            tasks: tasks,
        },

        // Begin training
        trained_model: {
            type: "microbert2.train::train",
            model: model,
            dataset_dict: { type: "ref", ref: "model_inputs" },
            training_engine: training_engine,
            log_every: 1,
            train_dataloader: train_dataloader,
            train_epochs: num_epochs,
            grad_accum: grad_accum,
            checkpoint_every: 100,
            validation_split: "dev",
            validation_dataloader: val_dataloader,
            val_metric_name: "mlm_perplexity",
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
