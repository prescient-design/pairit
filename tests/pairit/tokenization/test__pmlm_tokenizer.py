import importlib.resources

from torch import Size

from pairit.tokenization import PmlmTokenizer, PmlmTokenizerTransform


class TestPmlmTokenizer:
    def test_round_trip(self):
        inputs = ["ACDEFGHIKLMNPQRSTVWY"]

        path = importlib.resources.files("pairit") / "assets" / "pmlm_tokenizer"

        tokenizer = PmlmTokenizer.from_pretrained(path)
        tokenized_inputs = tokenizer(inputs)
        decoded_outputs = tokenizer.decode(
            token_ids=tokenized_inputs["input_ids"][0], skip_special_tokens=True
        ).replace(" ", "")

        assert inputs[0] == decoded_outputs

    def test_transform_shapes(self):
        inputs = 4 * ["ACDEFGHIKL"]

        transform_fn = PmlmTokenizerTransform(
            tokenizer_dir="pmlm_tokenizer",
            truncation=True,
            padding="max_length",
            max_length=32,
        )

        tokenized = transform_fn.transform(inputs, {})

        assert tokenized["input_ids"].shape == Size([4, 32])
        assert tokenized["labels"].shape == Size([4, 32])
