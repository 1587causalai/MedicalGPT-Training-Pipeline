import torch
import torch.nn as nn
from torch.nn import MSELoss
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from transformers import BloomModel, BloomPreTrainedModel, BloomConfig
from transformers.modeling_outputs import ModelOutput


@dataclass
class GaussianRewardModelOutput(ModelOutput):
    """
    Output class for the Gaussian Reward Model.
    """
    loss: Optional[torch.FloatTensor] = None
    mean: torch.FloatTensor = None
    variance: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BloomForGaussianRewardModeling(BloomPreTrainedModel):
    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.score = nn.Linear(config.hidden_size, 2, bias=False)
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], GaussianRewardModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        mean, log_var = logits.split(1, dim=-1)
        var = torch.exp(log_var)

        # 使用最后一个非填充token的输出
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = -1

        pooled_mean = mean[torch.arange(batch_size, device=mean.device), sequence_lengths].squeeze(-1)
        pooled_var = var[torch.arange(batch_size, device=var.device), sequence_lengths].squeeze(-1)

        loss = None
        if labels is not None:
            loss = -torch.distributions.Normal(pooled_mean, torch.sqrt(pooled_var)).log_prob(labels).mean()

        if not return_dict:
            output = (pooled_mean, pooled_var) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return GaussianRewardModelOutput(
            loss=loss,
            mean=pooled_mean,
            variance=pooled_var,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


# 使用示例
def main():
    config = BloomConfig.from_pretrained("bigscience/bloom-560m")
    config.num_labels = 1  # 为了兼容性，虽然我们不直接使用这个
    model = BloomForGaussianRewardModeling(config)

    # 假设的输入
    input_ids = torch.randint(0, 1000, (1, 10))
    labels = torch.randn(1)

    outputs = model(input_ids=input_ids, labels=labels)

    print(f"Loss: {outputs.loss}")
    print(f"Mean: {outputs.mean}")
    print(f"Variance: {outputs.variance}")


if __name__ == "__main__":
    main()