#!/usr/bin/python

def tensor_check(inputs, shape, max_val, min_val, dtype, arr_type=None):
    # test max value
    error_msg = "Error: the max value should be less than '{}', is '{}'".format(max_val, inputs.max())
    assert (inputs.max() <= max_val), error_msg
    # check min value
    error_msg = "Error: the min value should be bigger than '{}', is '{}'".format(min_val, inputs.min())
    assert (inputs.min() >= min_val), error_msg
    # check dtype
    error_msg = "Error: type should be '{}', is '{}'".format(dtype, inputs.dtype)
    assert (inputs.dtype == dtype), error_msg
    # check dtype
    if (arr_type is not None):
        error_msg = "Error: input type should be '{}', is '{}'".format(arr_type, type(inputs))
        assert (isinstance(inputs, arr_type)), error_msg
    #check shape
    error_msg = "Error: shape should be '{}', is '{}'".format(shape, inputs.shape)
    assert (inputs.shape == shape), error_msg
