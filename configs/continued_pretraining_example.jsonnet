// Example config: continued pretraining + NER + dependency parsing evals
//
// Usage:
//   tango run configs/continued_pretraining_example.jsonnet
//
// Replace the paths below with your actual data.

// Model to continue pretraining from (HuggingFace ID or local path)
# local base_model = "xlm-roberta-base";
local base_model = "jhu-clsp/mmBERT-base";

// Plain-text corpus for continued pretraining (one sentence per line)
local pretrain_text = "data/wo/train.txt";
local pretrain_dev_text = "data/wo/dev.txt";  // optional; set to null to skip

// Where to save the continued-pretrained base model
local pretrained_output = "workspace/continued_pretrain/xlm-r-wolof";

// NER data
local ner_train = "data/wo/train.bio";
local ner_dev   = "data/wo/dev.bio";
local ner_test  = "data/wo/test.bio";
local ner_save  = "workspace/ner/xlm-r-wolof";

// Dependency parsing data
local dep_train = "data/wo/train.conllu";
local dep_dev   = "data/wo/dev.conllu";
local dep_test  = "data/wo/test.conllu";
local dep_save  = "workspace/dep/xlm-r-wolof";

{
    steps: {
        // Step 1: Continued MLM pretraining
        continued_pretraining: {
            type: "microbert2.microbert.eval.continued_pretraining::continued_mlm_pretraining",
            model_name_or_path: base_model,
            train_data_path: pretrain_text,
            output_path: pretrained_output,
            dev_data_path: pretrain_dev_text,
            num_epochs: 3,
            batch_size: 8,
            gradient_accumulation_steps: 4,
            learning_rate: 2e-5,
        },

        // Step 2: NER eval on the continued-pretrained model
        evaluate_ner: {
            type: "microbert2.microbert.eval.ner::evaluate_ner",
            model_path: { type: "ref", ref: "continued_pretraining" },
            train_data_path: ner_train,
            dev_data_path: ner_dev,
            test_data_path: ner_test,
            save_path: ner_save,
        },

        // Step 3: Dependency parsing eval on the continued-pretrained model
        evaluate_dependency_parsing: {
            type: "microbert2.microbert.eval.dependency_parsing::evaluate_dependency_parsing",
            model_path: { type: "ref", ref: "continued_pretraining" },
            train_data_path: dep_train,
            dev_data_path: dep_dev,
            test_data_path: dep_test,
            save_path: dep_save,
        },
    }
}
