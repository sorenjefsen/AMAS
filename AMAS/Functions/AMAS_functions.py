import numpy as np
import matplotlib.pyplot as plt

#________________________________________________________________________________________________
#region Statistics
class Stat():
    def __init__(self, data, perform_all = False):
        self.data = data

    def mean(self):
        return np.mean(self.data)
    def variance(self):
        return np.var(self.data)
    def std_dev(self):
        return np.std(self.data)
    
    def basic_stats(self, display = True):
        if display:
            print('Mean:', self.mean())
            print('Variance:', self.variance())
            print('Standard Deviation:', self.std_dev())
        return self.mean(), self.variance(), self.std_dev() 
    
    def compare_with_function(self, function):
        chi_square = None
        return function(self.data)
    
def unbiased_variance(data, expected_value = None):
    n = len(data)
    if expected_value == None:
        expected_value = np.mean(data)
    return np.sum((data - expected_value)**2)/(n-1)

def chi_square(data, expected_value = None):
    """
    Chi-square function
    
    """
    if expected_value == None:
        expected_value = np.mean(data)
    return np.sum((data - expected_value)**2)/expected_value
#endregion
#________________________________________________________________________________________________
#region Quality of Life functions
def normalizer(func, interval = [0, 1], n = 1000, **kwargs):
    """
    Function for numerically normalizing a function.
    Parameters:
        func: function to be normalized
        interval: list of two floats, [a, b], the interval over which to normalize the function
        n: int, number of points to sample the function at
        **kwargs: parameters of the function to be normalized

    Returns:
        function_normalized: function, the normalized function
            The new functions takes a single argument, x, so the parameters of the original function is fixed upon normalization.

    """
    x = np.linspace(interval[0], interval[1], n)
    y = func(x, **kwargs)

    def function_normalized(x):
        return func(x, **kwargs) / max(y)
    
    return function_normalized



def knotter(x, y, N = None, knot_length = None):
    """ 
    Function that creates spline knots from data. Can take either the amount of knots, or the length of the knots. 

    Parameters:
        x: array, x-values of data
        y: array, y-values of data
        N: int, number of knots
        knot_length: float, length of knots
    """
    # Choose between using N or knot_length
    if knot_length == None:
        knot_length = (max(x) - x[0])/N

    # Initialize variables
    max_x = x[0] + knot_length
    result, temp_x, temp_y = [], [], []

    # Run through the data
    for i in range(len(x)):
        # Run through segment
        if x[i] < max_x:
            temp_x.append(x[i])
            temp_y.append(y[i])

        # If segment is done, append mean values to result, and go to next segment
        if x[i] > max_x:
            result.append([np.mean(temp_x), np.mean(temp_y)])
            temp_x = []
            temp_y = []
            max_x = x[i] + knot_length

    # Append last segment
    result.append([np.mean(temp_x), np.mean(temp_y)])


    return np.array(result).T


def spliner(x, y, N = 10, scale = None, kind = 'linear', return_function = False):
    """
    Function that creates splines from 1D dataset. 
    You can either specify the number of extra points betwen each datapoint,
    or you can specify a scale factor for the total number of points.

    Parameters:
        x: array, x-values of data
        y: array, y-values of data
        N: int, number of points between each data point
        scale: float, scale factor for the number of points

    Returns:
        result_x, result y: tuple of arrays, x and y values of the spline
        or
        f: function, the spline function, taking in x-values and returning y-values of spline
    """

    if kind == 'linear':
        # If we want the function, we need external library
        if return_function:
            from scipy.interpolate import interp1d
            f = interp1d(result_x, result_y, kind='linear')
            return f

        # Otherwise we use my own code for linear interpolation

        # Choose between N and scale
        if scale != None:
            N = int(len(x)*scale)
            
        result_y = np.array([])
        result_x = np.array([])
        for i in range(len(x)-1):
            # Get the two points
            x_0, x_1 = x[i], x[i+1]
            y_0, y_1 = y[i], y[i+1]

            # Calculate the slope and the line
            diff_1 = (y_1 - y_0)/(x_1 - x_0)
            x_out = np.linspace(x_0, x_1, N)
            y_out = y_0 + diff_1*(x_out - x_0)

            # Append to the result
            result_x = np.append(result_x, x_out)
            result_y = np.append(result_y, y_out)

        else:
            return result_x, result_y

    # For other interpolation methods, we use scipy
    if kind == 'cubic':
        from scipy.interpolate import CubicSpline
        f = CubicSpline(x, y)

    if kind == 'quadratic':
        from scipy.interpolate import interp1d
        f = interp1d(x, y, kind='quadratic')

    # And we choose between the function or the values
    if return_function:
        return f

    else:
        x_new = np.linspace(min(x), max(x), N)
        y_new = f(x_new)
        return x_new, y_new

#endregion
#________________________________________________________________________________________________
#region Monte Carlo
def MC_sampler(func, interval=[0, 1], n_points=1000, **kwargs):
    """
    Function for sampling a function using the Monte Carlo method. 
    Its semi-vectorized, making it very fast compared to a single loop. It makes rougly 22 loop steps at 1.000.000 points
    !! Works only for functions that are bounded in an interval !! !! No gaussians !!

    Parameters:
        func: function to sample
        interval: list of two floats, [a, b], the interval over which to sample the function
        n_points: int, number of points to sample
        **kwargs: parameters of the function to be sampled

    Returns:
        points: array of floats, the sampled points
    """
    count = 0
    n_step = n_points
    output = np.array([])
    while len(output) < n_points:
        x = np.random.uniform(interval[0], interval[1], n_step)
        y = np.random.uniform(0, 1, n_step)
        output = np.append(output, x[y < func(x, **kwargs)])
        n_step = n_points - len(output)
        count += 1

    print(count)
    return output

#endregion 
#________________________________________________________________________________________________

#region Function Expressions
def gaussian(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)

def likelihood(func, x, **kwargs):
    return np.prod(func(x, **kwargs))

def ln_likelihood(func, x, **kwargs):
    return np.sum(np.log(func(x, **kwargs)))

def neg_ln_likelihood(func, x, **kwargs):
    return -np.sum(np.log(func(x, **kwargs)))
#endregion
#________________________________________________________________________________________________


