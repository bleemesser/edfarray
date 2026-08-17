"""Pad behavior for 2D proxies over mixed-rate ("open") groups.

Reads past a channel's valid length are governed by `pad_mode`. The physical and digital paths
share one implementation, so both are exercised here.
"""

import edfarray
import numpy as np
import pytest
from conftest import FIXTURES

PATH = str(FIXTURES / "test_generator.edf")


@pytest.fixture
def mixed():
    """A group spanning two different sample rates, so shorter channels need padding."""
    f = edfarray.EdfFile(PATH)
    groups = f.signal_groups()
    by_rate = sorted(groups, key=len, reverse=True)
    indices = list(by_rate[0].indices[:2]) + list(by_rate[-1].indices[:1])
    group = f.signal_group(indices)
    assert group.kind == "open", "fixture no longer yields a mixed-rate group"
    return f, group


def test_raise_is_the_default(mixed):
    f, group = mixed
    proxy = f.proxy_2d(group)
    assert proxy.pad_mode == "raise"
    beyond = min(proxy.valid_lengths) + 1
    with pytest.raises(edfarray.OutOfRangeError):
        proxy[:, 0:beyond]


def test_within_valid_length_never_pads(mixed):
    f, group = mixed
    safe = min(f.proxy_2d(group).valid_lengths)
    data = f.proxy_2d(group)[:, 0:safe]
    assert not np.isnan(data).any()


def test_nan_pad(mixed):
    f, group = mixed
    proxy = f.proxy_2d(group, pad_mode="nan")
    short = int(np.argmin(proxy.valid_lengths))
    beyond = min(proxy.valid_lengths) + 10
    data = proxy[:, 0:beyond]
    assert np.isnan(data[short, -1])
    assert not np.isnan(data[int(np.argmax(proxy.valid_lengths)), -1])


def test_zero_pad(mixed):
    f, group = mixed
    proxy = f.proxy_2d(group, pad_mode="zero")
    short = int(np.argmin(proxy.valid_lengths))
    data = proxy[:, 0 : min(proxy.valid_lengths) + 10]
    assert data[short, -1] == 0.0


def test_numeric_pad_value(mixed):
    f, group = mixed
    proxy = f.proxy_2d(group, pad_mode=-7.5)
    short = int(np.argmin(proxy.valid_lengths))
    data = proxy[:, 0 : min(proxy.valid_lengths) + 10]
    assert data[short, -1] == -7.5


def test_edge_pad_repeats_last_valid_sample(mixed):
    f, group = mixed
    proxy = f.proxy_2d(group, pad_mode="edge")
    short = int(np.argmin(proxy.valid_lengths))
    valid = proxy.valid_lengths[short]
    data = proxy[:, 0 : valid + 5]
    last_real = data[short, valid - 1]
    assert data[short, valid] == last_real
    assert data[short, -1] == last_real


def test_unknown_pad_mode_rejected(mixed):
    f, group = mixed
    with pytest.raises(ValueError):
        f.proxy_2d(group, pad_mode="wrap")


def test_nan_pad_rejected_for_digital_reads(mixed):
    f, group = mixed
    proxy = f.proxy_2d(group, pad_mode="nan")
    beyond = min(proxy.valid_lengths) + 10
    # NaN has no int32 representation, so the digital path must refuse it rather than invent one.
    with pytest.raises(edfarray.InvalidArgumentError):
        proxy.read_digital(list(range(len(group))), 0, beyond)


def test_valid_lengths_match_channel_sample_counts(mixed):
    f, group = mixed
    proxy = f.proxy_2d(group)
    for pos, sig_idx in enumerate(group.indices):
        assert proxy.valid_lengths[pos] == len(f.signal(sig_idx))
