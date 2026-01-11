#!/usr/bin/python

def tensor_check(inputs, shape, max_val, min_val, dtype, arr_type=None):
    # check dtype
    error_msg = "Error: type should be '{}', is '{}'".format(dtype, inputs.dtype)
    assert (inputs.dtype == dtype), error_msg
    # check dtype
    if (arr_type is not None):
        error_msg = "Error: input type should be '{}', is '{}'".format(arr_type, type(inputs))
        assert (isinstance(inputs, arr_type)), error_msg
    # test max value
    error_msg = "Error: the max value should be less than '{}', is '{}'".format(max_val, inputs.max())
    assert (inputs.max() <= max_val), error_msg
    # check min value
    error_msg = "Error: the min value should be bigger than '{}', is '{}'".format(min_val, inputs.min())
    assert (inputs.min() >= min_val), error_msg
    #check shape
    if (shape is not None):
        if (None not in shape):
            error_msg = "Error: shape should be '{}', is '{}'".format(shape, inputs.shape)
            assert (inputs.shape == shape), error_msg
        else:
            error_msg = "Error: size of shape is different, should be '{}', is '{}'".format(shape, inputs.shape)
            assert (len(inputs.shape) == len(shape)), error_msg
            for benck_shape, real_shape in zip(shape, inputs.shape):
                if (benck_shape is not None):
                    error_msg = "Error: shape should be '{}', is '{}'".format(shape, inputs.shape)
                    assert (benck_shape == real_shape), error_msg

