# cost functions
import numpy
import pandas as pd

# causality
from causality.inference.search import IC
from causality.inference.independence_tests import RobustRegressionTest

# utilities
import plac


def main():
    # generate some toy data:
    size = 2000
    x1 = numpy.random.normal(size=size)
    x2 = x1 + numpy.random.normal(size=size)
    x3 = x1 + numpy.random.normal(size=size)
    x4 = x2 + x3 + numpy.random.normal(size=size)
    x5 = x4 + numpy.random.normal(size=size)

    # load the data into a dataframe:
    x = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})

    # define the variable types: 'c' is 'continuous'.  The variables defined here
    # are the ones the search is performed over  -- NOT all the variables defined
    # in the data frame.
    variable_types = {'x1': 'c', 'x2': 'c', 'x3': 'c', 'x4': 'c', 'x5': 'c'}

    # run the search
    ic_algorithm = IC(RobustRegressionTest, x, variable_types)
    graph = ic_algorithm.search()

    # view the edges
    for cause, effect, meta in graph.edges(data=True):
        print(cause, '->', effect, meta)


if __name__ == '__main__':
    try:
        plac.call(main)
    except KeyboardInterrupt:
        print('\nGoodbye!')
