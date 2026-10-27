from universal_jev.cli import build_parser


def test_training_cli_is_offline_by_default():
    args = build_parser().parse_args(
        ["--model", "local-model", "--data", "train.jsonl", "--output", "run"]
    )
    assert not args.allow_downloads
    assert args.epochs == 1
    assert args.learning_rate == 2e-4
