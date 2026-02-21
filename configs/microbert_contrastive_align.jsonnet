// --------------------------------------------------------------------------------
// Parameters
// --------------------------------------------------------------------------------
local language = "coptic";
// Optional and purely descriptive, intended to help you keep track of different model
// configurations. Set to `""` if you don't want to bother.
local experiment_name = "coptic_contrastive_align";

// Rclone Upload Configuration ---------------------------------------------------
// Set this to your rclone remote path to enable automatic upload after training
// Example: "myremote:path/to/models" where "myremote" is configured in rclone
// Set to null to disable upload
local rclone_remote_path = null;

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
// Parallel sentence pairs for contrastive alignment (tab-separated: src\ttgt)
local train_align_path = "data/cop/train.tsv";
local dev_align_path   = "data/cop/dev.tsv";
// local test_align_path  = "data/cop/test.tsv";  // uncomment if you have a test set

// Encoder ------------------------------------------------------------------------
local max_length = 512;
local hidden_size = 128;
local num_layers = 4;
// Type of encoder stack. See microbert2/microbert/model/encoder.py for implementations.
// local bert_type = "bert";
local bert_type = "modernbert";
// local bert_type = "electra";

// Encoder stack configuration.
// See https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertConfig
// Note that vocab_size will be overridden, so do not set it based on your tokenizer settings,
// so do not overwrite it.
local bert_config = {
    hidden_size: hidden_size,
    num_hidden_layers: num_layers,
    num_attention_heads: 8,
    intermediate_size: hidden_size + hidden_size / 2,
    max_position_embeddings: max_length,
    attention_dropout: 0.1,
    embedding_dropout: 0.1,
    mlp_dropout: 0.1,
    global_attn_every_n_layers: 1,
};

// Training and Optimization ------------------------------------------------------
local batch_size = 128;
local grad_accum = 1;
local effective_batch_size = grad_accum * batch_size;
local num_steps = 150000;
local validate_every = 5000;  // in steps

local optimizer = {
    type: "torch::AdamW",
    lr: 3e-3,
    betas: [0.9, 0.98],
    eps: 1e-6,
    weight_decay: 0.05
};

local lr_scheduler = {
    type: "transformers::reduce_lr_on_plateau",
    factor: 0.8,
    patience: 5,
    min_lr: 1e-5,
};

// When True, attempt to scale loss contribution from each task using learnable parameters
// See https://arxiv.org/abs/1705.07115
local loss_auto_scaling = false;

// Some set up, don't modify ------------------------------------------------------
local util = import 'lib/util.libsonnet';
local model_path = (
    "./workspace/models/" + language + "_" + experiment_name + "_" + util.stringifyObject(bert_config)
);
local tokenizer = { pretrained_model_name_or_path: model_path };

// Tasks --------------------------------------------------------------------------
local mlm_task = {
    type: "microbert2.microbert.tasks.mlm.MLMTask",
    dataset: { type: "ref", ref: "raw_text_data" },
    tokenizer: tokenizer,
};
local contrastive_align_task = {
    type: "microbert2.microbert.tasks.contrastive_align.ContrastiveAlignTask",
    train_path: train_align_path,
    dev_path: dev_align_path,
    // test_path: test_align_path,   // uncomment if you have a test set
    delimiter: "\t",
    mbart_tokenizer: "facebook/mbart-large-50-many-to-one-mmt",
    mbart_encoder_model: "facebook/mbart-large-50-many-to-one-mmt",
    src_lang_code: "ar_AR",
    tgt_lang_code: "en_XX",
    max_sequence_length: 128,
    proportion: 0.1,
    head: {
        num_encoder_layers: num_layers,
        embedding_dim: hidden_size,
        projection_dim: 256,
        temperature: 0.07,
        use_layer_mix: false,
        contrastive_weight: 1.0,
    },
};
local tasks = [mlm_task, contrastive_align_task];


// --------------------------------------------------------------------------------
// Internal--don't modify below here unless you're sure you know what you're doing!
// --------------------------------------------------------------------------------
local model = {
    type: "microbert2.microbert.model.model::microbert_model",
    tokenizer: tokenizer,
    model_output_path: model_path,
    loss_auto_scaling: loss_auto_scaling,
    tasks: tasks,
    encoder: {
        type: bert_type,
        tokenizer: tokenizer,
        bert_config: bert_config,
    }
};

local training_engine = {
    type: "mb2",
    optimizer: optimizer,
    lr_scheduler: lr_scheduler,
    amp: false,
    max_grad_norm: 10,
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
    //num_workers: 2,
    //prefetch_factor: 4,
    //persistent_workers: true,
};
local val_dataloader = {
    shuffle: false,
    batch_size: batch_size,
    collate_fn: collate_fn,
    pin_memory: true,
    //num_workers: 2,
    //prefetch_factor: 4,
    //persistent_workers: true,
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
            train_steps: num_steps,
            grad_accum: grad_accum,
            validate_every: validate_every,
            checkpoint_every: validate_every,
            validation_split: "dev",
            validation_dataloader: val_dataloader,
            val_metric_name: "mlm_perplexity",
            auto_aggregate_val_metric: false,
            callbacks: [
                {
                    type: "microbert2.microbert.model.model::write_model",
                    path: model_path,
                    model_attr: "encoder.encoder"
                },
                {type: "microbert2.microbert.model.model::reset_metrics"},
                {
                    type: "microbert2.microbert.model.model::rclone_upload",
                    remote_path: rclone_remote_path,
                    upload_logs: true
                }
            ],
        },
    }
}
