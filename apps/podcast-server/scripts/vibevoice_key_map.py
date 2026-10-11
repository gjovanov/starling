"""Key mapping of transformers' VibeVoice checkpoint converter, vendored verbatim.

Source: huggingface/transformers @ 640a08a597034221ca1c4fc0c129cf0118179225
        src/transformers/models/vibevoice/convert_vibevoice_to_hf.py
        (STATE_DICT_MAPPING + map_old_key_to_new; Apache-2.0, Copyright 2026
        The HuggingFace Team). Conversion scripts are not shipped in the
        transformers wheel, so scripts/verify_provenance.py carries its own copy.
"""

import re

# fmt: off
STATE_DICT_MAPPING = {
    # Semantic tokenizer encoder
    r"semantic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.":    r"semantic_tokenizer_encoder.stem.conv.conv.",
    r"semantic_tokenizer\.encoder\.stages\.0\.":                        r"semantic_tokenizer_encoder.stem.stage.",
    r"semantic_tokenizer\.encoder\.downsample_layers\.(\d+)\.0\.conv\.": r"semantic_tokenizer_encoder.conv_layers.PLACEHOLDER.conv.conv.",
    r"semantic_tokenizer\.encoder\.stages\.(\d+)\.":                    r"semantic_tokenizer_encoder.conv_layers.PLACEHOLDER.stage.",
    r"semantic_tokenizer\.encoder\.head\.conv\.":                       r"semantic_tokenizer_encoder.head.",

    # Acoustic tokenizer encoder
    r"acoustic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.":    r"audio_tower.encoder.stem.conv.conv.",
    r"acoustic_tokenizer\.encoder\.stages\.0\.":                        r"audio_tower.encoder.stem.stage.",
    r"acoustic_tokenizer\.encoder\.downsample_layers\.(\d+)\.0\.conv\.": r"audio_tower.encoder.conv_layers.PLACEHOLDER.conv.conv.",
    r"acoustic_tokenizer\.encoder\.stages\.(\d+)\.":                    r"audio_tower.encoder.conv_layers.PLACEHOLDER.stage.",
    r"acoustic_tokenizer\.encoder\.head\.conv\.":                       r"audio_tower.encoder.head.",

    # Acoustic tokenizer decoder: upsample_layers.0 -> stem, upsample_layers.N -> conv_layers.N-1
    r"acoustic_tokenizer\.decoder\.upsample_layers\.0\.0\.conv\.conv\.":           r"audio_tower.decoder.stem.conv.conv.",
    r"acoustic_tokenizer\.decoder\.stages\.0\.":                                   r"audio_tower.decoder.stem.stage.",
    r"acoustic_tokenizer\.decoder\.upsample_layers\.(\d+)\.0\.convtr\.convtr\.":  r"audio_tower.decoder.conv_layers.PLACEHOLDER.convtr.convtr.",
    r"acoustic_tokenizer\.decoder\.stages\.(\d+)\.":                               r"audio_tower.decoder.conv_layers.PLACEHOLDER.stage.",
    r"acoustic_tokenizer\.decoder\.head\.conv\.":                                  r"audio_tower.decoder.head.",

    # Rename any remaining acoustic tokenizer keys (the module is `audio_tower` in the HF model)
    r"acoustic_tokenizer\.":                                                       r"audio_tower.",

    # Diffusion head renaming
    r"prediction_head\.t_embedder\.mlp\.0\.":                                      r"diffusion_head.timestep_proj.fc1.",
    r"prediction_head\.t_embedder\.mlp\.2\.":                                      r"diffusion_head.timestep_proj.fc2.",
    r"prediction_head\.layers\.(\d+)\.adaLN_modulation\.1\.":                     r"diffusion_head.layers.\1.linear.",
    r"prediction_head\.final_layer\.adaLN_modulation\.1\.":                       r"diffusion_head.final_layer.linear_1.",
    r"prediction_head\.final_layer\.linear\.":                                     r"diffusion_head.final_layer.linear_2.",
    r"prediction_head\.":                                                          r"diffusion_head.",

    # Multimodal connectors (acoustic connector is the `multi_modal_projector` in the HF model)
    r"acoustic_connector\.fc1\.": r"multi_modal_projector.linear_1.",
    r"acoustic_connector\.norm\.": r"multi_modal_projector.act.",
    r"acoustic_connector\.fc2\.": r"multi_modal_projector.linear_2.",
    r"semantic_connector\.fc1\.": r"semantic_connector.linear_1.",
    r"semantic_connector\.norm\.": r"semantic_connector.act.",
    r"semantic_connector\.fc2\.": r"semantic_connector.linear_2.",

    # Latent factors
    r"^model\.speech_scaling_factor":                                              r"model.latent_scaling_factor",
    r"^model\.speech_bias_factor":                                                 r"model.latent_bias_factor",

    # Clean up nested conv layers (must be after above mappings)
    r"mixer\.conv\.conv\.conv\.":                                                  r"mixer.conv.",
    r"\.conv\.conv\.conv\.":                                                       r".conv.conv.",
}
# fmt: on


def map_old_key_to_new(old_key: str) -> str:
    new_key = old_key

    for pattern, replacement in STATE_DICT_MAPPING.items():
        match = re.search(pattern, new_key)
        if match:
            # Handle index shifts for conv_layers (downsample_layers/upsample_layers indexed from 1)
            if "PLACEHOLDER" in replacement and match.groups():
                layer_idx = int(match.group(1))
                # Shift down by 1 since layer 0 becomes stem
                new_idx = layer_idx - 1
                replacement = replacement.replace("PLACEHOLDER", str(new_idx))

            new_key = re.sub(pattern, replacement, new_key)

    return new_key
