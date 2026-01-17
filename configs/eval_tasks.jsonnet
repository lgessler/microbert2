// Data ---------------------------------------------------------------------------
// Using Coptic data as an example - replace with your own data paths
local train_conllu_path = "data/cop/cop_scriptorium-ud-train.conllu";
local dev_conllu_path = "data/cop/cop_scriptorium-ud-dev.conllu";
local test_conllu_path = "data/cop/cop_scriptorium-ud-test.conllu";

// Model ---------------------------------------------------------------------------
// Replace with your own model path  
local model = "workspace/models/microbert-coptic-mx";

// Path to save trained dependency parser model 
local save_path = "workspace/parsers/microbert-coptic-mx";

// Evaluation settings
local save_predictions = true;
local predictions_output = "workspace/parsers/microbert-coptic-mx/test_predictions.conllu";
local results_json = "workspace/parsers/microbert-coptic-mx/dependency_parsing_results.json";

{
    steps : {
        evaluate_dependency_parsing: {
            type: "microbert2.microbert.eval.dependency_parsing::evaluate_dependency_parsing",
            model_path: model,
            save_path: save_path,
            train_data_path: train_conllu_path,
            dev_data_path: dev_conllu_path,
            test_data_path: test_conllu_path,
        }
    }
}
