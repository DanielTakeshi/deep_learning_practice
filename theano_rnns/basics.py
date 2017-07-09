"""
The basics of Theano.
"""

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import time


def logistic_function():
    x = T.dmatrix('x')
    s = 1 / (1 + T.exp(-x))
    logistic = theano.function([x], s)
    print(logistic([[0,1],[-1,-2]]))


def multiple_arguments():
    a, b = T.dmatrices('a', 'b')
    diff = a - b
    abs_diff = abs(diff)
    diff_squared = diff**2
    f = theano.function([a, b], [diff, abs_diff, diff_squared])
    in_a = np.array([[1,1],[1,1]])
    in_b = np.array([[0,1],[2,3]])
    print(f(in_a,in_b))


def shared_variables():
    state = theano.shared(value=0)
    inc = T.iscalar('inc')
    accumulator = theano.function([inc], state, updates=[(state, state+inc)])
    decrementor = theano.function([inc], state, updates=[(state, state-inc)])
    print(state.get_value()) # 0
    accumulator(1)
    print(state.get_value()) # 1
    accumulator(10)
    print(state.get_value()) # 11
    state.set_value(-1)
    print(state.get_value()) # -1
    decrementor(2)
    print(state.get_value()) # -3


def random_numbers():
    from theano.tensor.shared_randomstreams import RandomStreams
    srng = RandomStreams(seed=234)
    # Only for the CPU.
    rv_u = srng.uniform((2,2))
    rv_n = srng.normal((2,2))
    f = theano.function([], rv_u)
    g = theano.function([], rv_n, no_default_updates=True)
    nearly_zeros = theano.function([], rv_u+rv_u - 2*rv_u)
    print(nearly_zeros())


def logistic_regression():
    N = 400
    feats = 784
    lambdareg = 0.01

    # The synthetic dataset, D = (input_values, target_class).
    D = (np.random.randn(N,feats), np.random.randint(size=N, low=0, high=2))
    training_steps = 5000

    # Theano symbolic variables.
    x = T.dmatrix('x')
    y = T.dvector('y')

    # The weight and bias are _shared_ so they can be trained.
    w = theano.shared(np.random.randn(feats), name='w')
    b = theano.shared(0., name='b')
    print("\nInitial model:")
    print("w:\n{}".format(w.get_value()))
    print("b:\n{}".format(b.get_value()))

    # Construct the Theano expression graph, p_1 = prob that the target is 1.
    p_1 = 1 / (1 + T.exp(-T.dot(x,w)-b))
    prediction = p_1 > 0.5
    cross_entropy_loss = -y*T.log(p_1) - (1-y)*T.log(1-p_1)
    cost = cross_entropy_loss.mean() + lambdareg * (w**2).sum()
    gw,gb = T.grad(cost=cost, wrt=[w,b]) # Computes both gradients w.r.t. the cost.

    # Compile everything, the previous stuff is just constructing the graph.
    # Note also that we're explicitly providing the gradient here (well, not
    # "explicitly" since we didn't explicitly compute gw and gb but you know
    # what I mean).
    train = theano.function(
            inputs=[x,y],
            outputs=[prediction, cross_entropy_loss],
            updates=((w, w-0.1*gw), (b, b-0.1*gb))
    )
    predict = theano.function(inputs=[x], outputs=prediction)

    # Train and print.
    for i in range(training_steps):
        if i % 1000 == 0:
            print(i)
        pred, err = train(D[0], D[1])

    print("Final model:")
    print(w.get_value())
    print(b.get_value())
    print("target for D:")
    print(D[1])
    print("(final) predictions for D:")
    print(predict(D[0]))
    print(np.sum(D[1] == predict(D[0])))
    print("")


def computing_gradients():
    # To be clear, T.grad(cost=s, wrt=w) requires s to be a _scalar_ expression.
    # It will also be efficiently computed since the returned expression is
    # optimized during compilation. This is an example of "symbolic
    # differentiation" where we can explicitly find the derivative, but more
    # generally, Theano technically does "automatic differentiation" because I
    # think their way isn't what people would normally call "symbolic
    # differentiation" but the terminology is so confusing here. :-(
    # https://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/
    from theano import pp
    x = T.dscalar('x')
    y = x ** 2
    gy = T.grad(cost=y, wrt=x) # gradient of x (2nd arg) w.r.t. function y (1st arg).
    print(pp(gy)) # will look like (2) * x^(2-1) which is correct. :-)
    f = theano.function(inputs=[x], outputs=gy) # Literally return the gradient.
    print(f(4))
    print(np.allclose(f(94.2), 188.4))


def other_gradients():
    # Jacobian and Hessian blah blah blah.
    # Just call their methods. But the docs say the `scan` method might be used
    # for recurrent computations. What about RNNs? I don't know.
    pass


def test_conditions():
    # Might be useful for me if I need to handle different cases. Not sure.
    # PS: the docs say that the plural constructors (for making TensorVariable
    # objects) aren't typically used in practice, just in tutorials to save
    # space. I agree, I like having more fine-grained control, it might be
    # needed. Note that it is NOT needed to explicitly state the dtype to be the
    # float value in the configuration. See:
    # http://deeplearning.net/software/theano/library/tensor/basic.html#creation
    a,b = T.scalars('a','b')
    x,y = T.matrices('x','y')

    # T.lt = "less than."
    z_switch = T.switch(T.lt(a,b), T.mean(x), T.mean(y))
    z_lazy = ifelse(T.lt(a,b), T.mean(x), T.mean(y))
    f_switch = theano.function(inputs=[a,b,x,y], outputs=z_switch,
            mode=theano.Mode(linker='vm'))
    f_lazyifelse = theano.function(inputs=[a,b,x,y], outputs=z_lazy,
            mode=theano.Mode(linker='vm'))

    val1 = 0.
    val2 = 1.
    big_mat1 = np.ones((10000,1000)).astype(theano.config.floatX)
    big_mat2 = np.ones((10000,1000)).astype(theano.config.floatX)
    n_times =10

    # Eh just a timing experiment.
    tic = time.clock()
    for i in range(n_times):
        f_switch(val1, val2, big_mat1, big_mat2)
    print("time both {}".format(time.clock()-tic))
    tic = time.clock()
    for i in range(n_times):
        f_lazyifelse(val1, val2, big_mat1, big_mat2)
    print("time one {}".format(time.clock()-tic))


if __name__ == "__main__":
    # Other stuff that might interest me are the shape information:
    # http://deeplearning.net/software/theano/tutorial/shape_info.html
    logistic_function()
    multiple_arguments()
    shared_variables()
    random_numbers()
    logistic_regression()
    computing_gradients()
    other_gradients()
    test_conditions()
