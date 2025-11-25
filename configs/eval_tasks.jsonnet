// Data ---------------------------------------------------------------------------
// Using Coptic data as an example - replace with your own data paths
local train_conllu_path = "../slate/mlt/mt_mudt-ud-train.conllu";
local dev_conllu_path = "../slate/mlt/mt_mudt-ud-dev.conllu";
local test_conllu_path = "../slate/mlt/mt_mudt-ud-test.conllu";

// Model ---------------------------------------------------------------------------
// Replace with your own model path  
local model = "workspace/models/maltese_maltese_mlm_mt_platue_200K_attention_dropout-0.1_embedding_dropout-0.1_global_attn_every_n_layers-1_hidden_size-128_intermediate_size-192_max_position_embeddings-512_mlp_dropout-0.1_num_attention_heads-8_num_hidden_layers-4";

// Path to save trained dependency parser model 
local save_path = "workspace/parsers/maltese_mlm_mt_platue_200K";

// Evaluation settings
local save_predictions = true;
local predictions_output = "workspace/parsers/maltese_mlm_mt_platue_200K/test_predictions.conllu";
local results_json = "workspace/parsers/maltese_mlm_mt_platue_200K/dependency_parsing_results.json";

{
    steps : {
        evaluate_dependency_parsing: {
            type: "microbert2.microbert.eval.dependency_parsing::evaluate_dependency_parsing",
            model_path: model,
            save_path: save_path,
            train_data_path: train_conllu_path,
            dev_data_path: dev_conllu_path,
            test_data_path: test_conllu_path,
            predictions_output: predictions_output,
            results_json: results_json,
        }
    }
}
