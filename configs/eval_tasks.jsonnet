// Data ---------------------------------------------------------------------------
// Dependency Parsing data (CoNLL-U format)
local train_conllu_path = "data/cop/cop_scriptorium-ud-train.conllu";
local dev_conllu_path = "data/cop/cop_scriptorium-ud-dev.conllu";
local test_conllu_path = "data/cop/cop_scriptorium-ud-test.conllu";

// NER data (BIO format) - replace with your own data paths
local train_ner_path = "data/ner/train.bio";
local dev_ner_path = "data/ner/dev.bio";
local test_ner_path = "data/ner/test.bio";

// Model ---------------------------------------------------------------------------
// Replace with your own model path
local model_path = "workspace/models/microbert-coptic-mx";

// Path to save trained dependency parser model
local parser_save_path = "workspace/parsers/microbert-coptic-mx";

// Path to save trained NER model
local ner_save_path = "workspace/ner/microbert-coptic-mx";

{
    steps : {
        // Dependency Parsing Evaluation
        evaluate_dependency_parsing: {
            type: "microbert2.microbert.eval.dependency_parsing::evaluate_dependency_parsing",
            model_path: model_path,
            save_path: parser_save_path,
            train_data_path: train_conllu_path,
            dev_data_path: dev_conllu_path,
            test_data_path: test_conllu_path,
        },

        // NER Evaluation
        evaluate_ner: {
            type: "microbert2.microbert.eval.ner::evaluate_ner",
            model_path: model_path,
            train_data_path: train_ner_path,
            dev_data_path: dev_ner_path,
            test_data_path: test_ner_path,
            save_path: ner_save_path,
            predictions_output: ner_save_path + "/predictions_test.bio",
            results_json: ner_save_path + "/results.json",
            batch_size: 16,
            learning_rate: 5e-5,
            num_epochs: 3,
        }
    }
}
