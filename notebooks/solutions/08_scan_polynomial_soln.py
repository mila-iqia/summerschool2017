import numpy

import theano
import theano.tensor as tt

coefficients = tt.vector("coefficients")
x = tt.scalar("x")
max_coefficients_supported = 10000

# Generate the components of the polynomial
full_range = tt.arange(max_coefficients_supported)


outputs_info = tt.as_tensor_variable(numpy.asarray(0, 'float64'))

components, updates = theano.scan(
    fn=lambda coeff, power, prior_value, free_var:
    prior_value + (coeff * (free_var ** power)),
    sequences=[coefficients, full_range],
    outputs_info=outputs_info,
    non_sequences=x)

polynomial = components[-1]
calculate_polynomial = theano.function(
    inputs=[coefficients, x],
    outputs=polynomial, updates=updates)

test_coeff = numpy.asarray([1, 0, 2], dtype=numpy.float32)
print calculate_polynomial(test_coeff, 3)
# 19.0
