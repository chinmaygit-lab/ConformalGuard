from conformalguard.cli import build_parser


def test_demo_parser_accepts_compact_grid_options():
    args = build_parser().parse_args(
        [
            "demo",
            "--seeds",
            "1,2,3",
            "--covariate-severities",
            "0,0.5,1",
            "--concept-severities",
            "0,0.25,1",
        ]
    )
    assert args.command == "demo"
    assert args.seeds == (1, 2, 3)
    assert args.covariate_severities == (0.0, 0.5, 1.0)
    assert args.concept_severities == (0.0, 0.25, 1.0)


def test_benchmark_parser_requires_target():
    parser = build_parser()
    try:
        parser.parse_args(["benchmark", "data.csv"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("benchmark should require --target")


def test_parser_has_no_command_by_default():
    args = build_parser().parse_args([])
    assert args.command is None
