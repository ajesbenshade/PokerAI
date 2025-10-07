from nextgen.selfplay import SelfPlayConfig, parallel_simulate_dataset


def test_parallel_fallback_small():
    # Works whether ray is present or not; function falls back to serial
    cfg = SelfPlayConfig(max_hands=6)
    X, y = parallel_simulate_dataset(total_hands=6, cfg=cfg, workers=2, agent_kind="datagen")
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 512
    assert y.shape[0] > 0

