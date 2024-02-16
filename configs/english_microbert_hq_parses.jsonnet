// --------------------------------------------------------------------------------
// Language settings
// --------------------------------------------------------------------------------
local language = std.extVar("LANGUAGE");
local experiment_name = std.extVar("NAME");
local use_phrase = std.parseInt(std.extVar("PHRASE")) != 0;
local use_tree = std.parseInt(std.extVar("TREE")) != 0;
local use_sla = std.parseInt(std.extVar("SLA")) != 0;
local language_code_index = import 'lib/language_code.libsonnet';
local stanza_do_not_retokenize = import 'lib/stanza_do_not_retokenize.libsonnet';
local stanza_no_mwt = import 'lib/stanza_no_mwt.libsonnet';

// a helper
local stringifyPair(k,v) = std.toString(k) + "-" + std.toString(v);
local stringifyObject(o) = std.join('_', std.objectValues(std.mapWithKey(stringifyPair, o)));

// --------------------------------------------------------------------------------
// Model settings
// --------------------------------------------------------------------------------
local max_length = 512;

local use_parser = false;
local use_xpos = false;
local FROM_PRETRAINED = false;
local hidden_size = 128;
local num_layers = 3;
local bert_config = {
    hidden_size: hidden_size,
    num_hidden_layers: num_layers,
    num_attention_heads: 8,
    intermediate_size: 512,
    max_position_embeddings: max_length,
};
local model_path = "./workspace/models/" + language + "_" + experiment_name + "_"+ stringifyObject(bert_config);
local tokenizer = { pretrained_model_name_or_path: model_path };
local tree_sgcl_config = if !use_tree then null else {
    subtree_sampling_method: {type: "all"},
};
local phrase_sgcl_config = if !use_phrase then null else {};
local sla_config = if !use_sla then null else {max_distance: 4};
local model = {
    type: "microbert2.sgcl.model.model::sgcl_model",
    parser: if use_parser then parser else null,
    xpos_tagging: use_xpos,
    tokenizer: tokenizer,
    counts: { "type": "ref", "ref": "counts" },
    model_output_path: model_path,
    tree_sgcl_config: tree_sgcl_config,
    phrase_sgcl_config: phrase_sgcl_config,
    sla_config: sla_config,
    encoder: {
        type: "bert",
        tokenizer: tokenizer,
        bert_config: bert_config,
    }
};

// --------------------------------------------------------------------------------
// Trainer settings
// --------------------------------------------------------------------------------
// BERT's original settings:
//    We train with batch size of 256 sequences
//    (256 sequences * 512 tokens = 128,000 tokens/batch)
//    for 1,000,000 steps, which is approximately 40
//    epochs over the 3.3 billion word corpus.
local BERT_batch_size = 256;
local BERT_steps = 1e6;
local BERT_total_instances = BERT_steps * BERT_batch_size;

// our settings
// We want a batch size of 256 (standard in the TLM lit and shown to have benefits) but the GPU memory
// on machines I have can't handle more than 32 reliably. To get around this, use gradient accumulation
// for an effective batch size of 256. See:
// https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
local batch_size = 32;
local grad_accum = 8;
local effective_batch_size = grad_accum * batch_size;
// We do not need to correct by (BERT_batch_size / batch_size) in order to ensure we're getting through the
// same number of training instances because each step goes through `grad_accum` microbatches
local num_steps = BERT_steps / 16;  // 16 is an extra reduction we're making

local validate_every = 10000;

// --------------------------------------------------------------------------------
// Optimizer settings
// --------------------------------------------------------------------------------
local training_engine = {
    type: "torch",
    optimizer: {
        type: "torch::AdamW",
        lr: 1e-4,
        betas: [0.9, 0.999],
        eps: 1e-6,
        weight_decay: 0.01
    },
    lr_scheduler: {
        type: "transformers::cosine",
        num_warmup_steps: 24000,
        num_training_steps: num_steps,
    },
    amp: false
};

local collate_fn = {
    type: "microbert2.sgcl.collator::collator",
    tokenizer: tokenizer,
    tree_config: tree_sgcl_config,
    phrase_config: phrase_sgcl_config,
    sla_config: sla_config,
    // whether to replace [MASK] with 10% UNK and 10% random. should be true for electra, false for bert
    mask_only: false,
};
local train_dataloader = {
    shuffle: true,
    batch_size: batch_size,
    collate_fn: collate_fn,
    pin_memory: true,
    num_workers: 4,
    prefetch_factor: 4,
    persistent_workers: true,
};
local val_dataloader = {
    shuffle: false,
    batch_size: batch_size,
    collate_fn: collate_fn,
    pin_memory: true,
    num_workers: 4,
    prefetch_factor: 4,
    persistent_workers: true,
};

{
    steps: {
        parsed_text_data: {
            type: "microbert2.data.conllu::read_conllu",
            train_path: "data/hqp/train.conllu",
            dev_path: "data/hqp/dev.conllu",
        },

        // Train tokenizer if necessary
        [if FROM_PRETRAINED then null else "tokenizer"]: {
            type: "microbert2.data.tokenize::train_tokenizer",
            dataset: { "type": "ref", "ref": "parsed_text_data" },
            model_path: model_path,
        },

        // Tokenize input data
        tokenized_text_data: {
            type: "microbert2.data.tokenize::tokenize_plus",
            dataset: { "type": "ref", "ref": "parsed_text_data" },
            max_length: max_length,
            tokenizer: tokenizer,
            step_extra_dependencies: if FROM_PRETRAINED then [] else [ {type: "ref", "ref": "tokenizer" } ]
        },

        // Postprocess
        postprocessed_text_data: {
            type: "microbert2.data.postprocess::expand_trees_with_subword_edges",
            dataset: { type: "ref", ref: "tokenized_text_data" }
        },

        // Merge inputs
        model_inputs: {
            type: "microbert2.sgcl.data::finalize",
            dataset: { "type": "ref", "ref": "postprocessed_text_data" },
        },

        // Record label counts
        counts: {
            type: "microbert2.data.util::count_unique_values",
            dataset: { "type": "ref", "ref": "model_inputs" },
            keys: ["xpos", "deprel"],
        },

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
                {"type": "microbert2.model::write_model", path: model_path, model_attr: "encoder.encoder"}
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
