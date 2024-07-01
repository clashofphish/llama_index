import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.sagemaker_endpoint import SageMakerLLM


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in SageMakerLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@pytest.mark.asyncio()
async def test_async_sg_client():
    llm_ep = "meta-llama-3"
    profile_name = "default"
    region_name = "us-east-1"

    sg_model = SageMakerLLM(
        endpoint_name=llm_ep,
        profile_name=profile_name,
        region_name=region_name,
        temperature=0.3,
        model_kwargs={"max_new_tokens": 2048},
        system_prompt="You are an AI assistant who is excited to great the world!",
    )
    resp = await sg_model.acomplete("Hello, how are you?")
    print(resp)
