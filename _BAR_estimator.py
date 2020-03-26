"""
Functions the implement BAR estimator
"""
import numpy as np
from scipy.stats import iqr
from scipy.stats import norm


def dict_to_list(dict_of_list):
    """
    :param dict_of_list: dict: varname --> ndarray
    :return: list of dic: [ {varname: float, ...}, ...  ]
    """
    keys = list(dict_of_list.keys())
    key0 = keys[0]
    for key in keys[1:]:
        assert len(dict_of_list[key0]) == len(dict_of_list[key]), key0 + " and " + key + " do not have same len."

    n = len(dict_of_list[key0])
    ls_of_dic = []
    for i in range(n):
        dic = {key: dict_of_list[key][i] for key in keys}
        ls_of_dic.append(dic)
    return ls_of_dic


def get_values_from_trace(model, trace, burn=0):
    """
    :param model: pymc3 model
    :param trace: pymc3 trace object
    :param burn: int, number of steps to exclude
    :return: dict: varname --> ndarray
    """
    varnames = [var.name for var in model.vars]
    trace_values = {var: trace.get_values(var, burn=burn) for var in varnames}
    return trace_values


def std_from_iqr(data):
    return iqr(data) / 1.35


def fit_normal(x, sigma_robust=False):
    mu, sigma = norm.fit(x)
    if sigma_robust:
        sigma = std_from_iqr(x)
    res = {"mu": mu, "sigma": sigma}
    return res


def fit_normal_trace(trace_values, sigma_robust=False):
    res = {varname: fit_normal(trace_values[varname], sigma_robust=sigma_robust) for varname in trace_values}
    return res


def draw_normal_samples(mu_sigma_dict, nsamples, random_state=None):
    rand = np.random.RandomState(random_state)
    keys = mu_sigma_dict.keys()
    samples = {k: rand.normal(loc=mu_sigma_dict[k]["mu"], scale=mu_sigma_dict[k]["sigma"], size=nsamples)
               for k in keys}
    return samples


def log_normal_pdf(mu, sigma, y):
    sigma2 = sigma * sigma
    res = - 0.5 * np.log(2 * np.pi * sigma2) - (0.5 / sigma2) * (y - mu) ** 2
    return res


def log_normal_trace(trace_val, mu_sigma_dict):
    keys = list(trace_val.keys())
    if len(keys) == 0:
        return 0.

    k0 = keys[0]
    for k in keys[1:]:
        assert len(trace_val[k0]) == len(trace_val[k]), k0 + " and " + k + " do not have same len."

    nsamples = len(trace_val[k0])
    logp = np.zeros(nsamples, dtype=float)
    for k in keys:
        mu = mu_sigma_dict[k]["mu"]
        sigma = mu_sigma_dict[k]["sigma"]
        y = trace_val[k]
        logp += log_normal_pdf(mu, sigma, y)

    return logp


def fit_uniform(x, d=1e-20):
    lower = x.min() - d
    upper = x.max() + d
    res = {"lower": lower, "upper": upper}
    return res


def fit_uniform_trace(trace_values):
    res = {varname: fit_uniform(trace_values[varname]) for varname in trace_values}
    return res


def draw_uniform_samples(lower_upper_dict, nsamples, random_state=None):
    rand = np.random.RandomState(random_state)
    keys = lower_upper_dict.keys()
    samples = {k: rand.uniform(low=lower_upper_dict[k]["lower"],
                               high=lower_upper_dict[k]["upper"],
                               size=nsamples)
               for k in keys}
    return samples


def log_uniform_pdf(lower, upper, y):
    logp = np.zeros_like(y)
    logp[:] = -np.inf
    logp[(y >= lower) & (y <= upper)] = np.log(1. / (upper - lower))
    return logp


def log_uniform_trace(trace_val, lower_upper_dict):
    keys = list(trace_val.keys())
    k0 = keys[0]
    for k in keys[1:]:
        assert len(trace_val[k0]) == len(trace_val[k]), k0 + " and " + k + " do not have same len."

    nsamples = len(trace_val[k0])
    logp = np.zeros(nsamples, dtype=float)
    for k in keys:
        lower = lower_upper_dict[k]["lower"]
        upper = lower_upper_dict[k]["upper"]
        y = trace_val[k]
        logp += log_uniform_pdf(lower, upper, y)

    return logp