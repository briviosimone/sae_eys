import pytest
import numpy as np
import torch

from utils import (
    split_data, save_analysis, load_analysis, is_not_decreasing
)



def test_split_data():
    """
    This is to verify that utils.split_data works correctly, namely:
    1) The output dimensions are as expected;
    2) The test overlapping exception is correctly raised.
    """

    # Config
    np.random.seed(1)
    torch.manual_seed(1)

    # Generate random data
    ns = 140 
    nh = 513
    snapshots = torch.rand(ns, nh)

    # Checks correctness of dimensions with "correct splitting"
    utrain, uval, utest = split_data(
        snapshots = snapshots,
        split = [100, 20, 20]
    )
    assert utrain.shape == (100, nh)
    assert uval.shape == (20, nh)
    assert utest.shape == (20, nh)

    # Checks that an exception is raise with test overlapping
    with pytest.raises(ValueError):
        utrain, uval, utest = split_data(
            snapshots = snapshots,
            split = [100, 20, 25]
        )


def test_save_load_analysis():
    """
    This is o verify the correct functionality of function to save and load the 
    results of the analyses, namely, we test that the original and 
    saved-then-loaded dictionary exactly coincide.
    """

    # Create dictionary with different types of values
    orig_dict = {
        'a' : 1,
        'b' : np.array([3,4]),
        'c' : 'feature'
    }

    # Save then load
    filename = 'test_dict.obj'
    save_analysis(analysis_dict = orig_dict, filename = filename)
    loaded_dict = load_analysis(filename = filename)

    # Check
    assert orig_dict.keys() == loaded_dict.keys()
    for key in orig_dict.keys():
        assert np.prod(orig_dict[key] == loaded_dict[key])


def test_is_not_decreasing():
    """
    This is to verify that utils.is_not_decreasing recognizes decreasing and not
    decreasing sequences.
    """

    assert not is_not_decreasing([10, 5, 3, 2])
    assert is_not_decreasing([10, 20, 3])
