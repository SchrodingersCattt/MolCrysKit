import pytest

from molcrys_kit.chemistry import Embedding


def test_embedding_uses_stable_id_index_and_periodic_image_vector():
    embedding = Embedding(
        (("A", (0.1, 0.2, 0.3)), ("B", (0.9, 0.2, 0.3))),
        cell_A=((2.0, 0.0, 0.0), (0.5, 2.0, 0.0), (0.0, 0.0, 3.0)),
    )
    assert embedding.position("B") == (0.9, 0.2, 0.3)
    assert embedding.vector("A", "B", (1, 0, 0)) == pytest.approx((2.8, 0.0, 0.0))
    with pytest.raises(KeyError):
        embedding.position("missing")
