import torch
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, PreTrainedTokenizer


def load_bart_model(model_name: str):
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return model


def compute_alignment_loss(
    model,
    original_inputs,
    debiased_inputs,
    loss_type="mse"
):
    """
    original_inputs & debiased_inputs: Dict with keys 'input_ids' and 'attention_mask'
    Returns a scalar alignment loss between the two model outputs.
    """
    # forward pass (no decoder_input_ids needed here since we compare encoder->decoder output directly)
    output_orig = model(**original_inputs, output_hidden_states=True, return_dict=True)
    output_debiased = model(**debiased_inputs, output_hidden_states=True, return_dict=True)

    # compare logits or decoder hidden states
    if loss_type == "mse":
        loss = F.mse_loss(output_orig.logits, output_debiased.logits.detach())
    elif loss_type == "kl":
        loss = F.kl_div(
            F.log_softmax(output_orig.logits, dim=-1),
            F.softmax(output_debiased.logits.detach(), dim=-1),
            reduction="batchmean"
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return loss
