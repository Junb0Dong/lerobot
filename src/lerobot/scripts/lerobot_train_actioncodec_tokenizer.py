"""CLI for independently training the ActionCodec semantic tokenizer."""

from lerobot.actioncodec.trainer import SemanticTokenizerTrainConfig, train_semantic_tokenizer
from lerobot.configs import parser
from lerobot.utils.utils import init_logging


@parser.wrap()
def train_tokenizer(cfg: SemanticTokenizerTrainConfig) -> None:
    train_semantic_tokenizer(cfg)


def main() -> None:
    init_logging()
    train_tokenizer()


if __name__ == "__main__":
    main()
