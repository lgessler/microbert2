{
  "regexes": [
    [".*projection.*weight", {"type": "xavier_uniform"}],
    [".*projection.*bias", {"type": "zero"}],
    [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
    [".*tag_bilinear.*bias", {"type": "zero"}],
    [".*weight_ih.*", {"type": "xavier_uniform"}],
    [".*weight_hh.*", {"type": "orthogonal"}],
    [".*bias_ih.*", {"type": "zero"}],
    [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
  ]
}
