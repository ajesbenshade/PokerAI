from nextgen.selfplay import SelfPlayConfig, simulate_dataset


def test_simulate_dataset_small():
    X, y = simulate_dataset(hands=5, cfg=SelfPlayConfig(max_hands=5), agent_kind="datagen")
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 512
    assert y.shape[0] > 0

