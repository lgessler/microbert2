function (hidden_size, num_layers) {
    input_dim: hidden_size,
    num_layers: num_layers + 1,
    pos_tag_embedding_dim: 64,
    encoder: {
      type: "pytorch_transformer",
      input_dim: hidden_size + 64,
      num_layers: 1,
      feedforward_hidden_dim: 512,
      num_attention_heads: 8,
      positional_encoding: "sinusoidal",
      positional_embedding_size: hidden_size + 64,
    },
    //{ type: "passthrough", "input_dim": hidden_size + 64 },
    //{
    //  "type": "stacked_bidirectional_lstm",
    //  "input_size": hidden_size + 64,
    //  "hidden_size": hidden_size * 2,
    //  "num_layers": 2,
    //  "recurrent_dropout_probability": 0.3,
    //  "use_highway": true
    //},
    tag_representation_dim: 50,
    arc_representation_dim: 50,
    use_layer_mix: false,
    initializer: import "parser_initializer.libsonnet",
}