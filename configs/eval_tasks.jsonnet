

// Data ---------------------------------------------------------------------------
local train_conllu_path = "../slate/mlt/mt_mudt-ud-train.conllu";
local dev_conllu_path = "../slate/mlt/mt_mudt-ud-dev.conllu";
local test_conllu_path = "../slate/mlt/mt_mudt-ud-test.conllu";

// Model ---------------------------------------------------------------------------
// Path to your trained model or pretrained model name
// Examples: "en_ewt-electra", "biaffine-dep-en", or path to custom model
local model = "en_ewt-electra";
//path to save trained diaperser model
local save_path = ""
// Evaluation settings
local save_predictions = true;
local predictions_output = "../slate/mlt/mt_mudt-ud-test_predictions.conllu";
local results_json = "dependency_parsing_results.json";

{
    steps : {
        evaluate_dependency_parsing: {
            type: "microbert2.microbert.eval.dependency_parsing::evaluate_dependency_parsing",
            model_path: model,
            save_path: save_path
            test_data_path: test_conllu_path,
            save_predictions: save_predictions,
            predictions_output: predictions_output,
            device: device,
            save_results_json: results_json,
        }
    }
}