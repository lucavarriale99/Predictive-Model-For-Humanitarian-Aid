from typing import Union, Callable, Iterable, Optional, Any, Tuple, List, Hashable
from tqdm import tqdm
import numpy as np
import pandas as pd
import xgboost as xgb
import networkx as nx
from abc import ABC, abstractmethod
from copy import copy
from scipy.stats import gaussian_kde, iqr
from sklearn.decomposition import PCA


class BaseKernel(ABC):

    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, kernel_functions: Optional[Iterable[Callable]], kernel_coefficients: Optional[Union[Iterable[float], str]] = None):
        raise NotImplementedError(f'Fit must be implemented')
    
    @abstractmethod
    def predict(self, predt: np.ndarray):
        raise NotImplementedError(f'Predict must be implemented')

    def infer_kernel_coefficients_from_data(self, X: Union['pd.DataFrame',np.ndarray]) -> List[Callable]:
        types = X.dtypes
        X = np.array(X)
        coeffs = []
        for i, t in enumerate(types):
            if pd.api.types.is_integer_dtype(t):
                coeffs.append(1)
            elif pd.api.types.is_float_dtype(t):
                coeffs.append(1.06*np.std(X[:,i])*(len(X))**(-0.2)) # Silverman rule of thumb
            else:
                raise TypeError(f'{t} not supported {X[:,i]}')
        return coeffs

    def infer_kernel_from_data(self, X: Union['pd.DataFrame',np.ndarray]) -> List[Callable]:
        types = X.dtypes
        kfs = []
        for i, t in enumerate(types):
            if pd.api.types.is_integer_dtype(t):
                kfs.append(np.vectorize(lambda x: 1 if x==0 else 0 ))
            elif pd.api.types.is_float_dtype(t):
                kfs.append(lambda x: np.power(2*np.pi, -0.5)*np.exp(-np.power(x, 2)/2))
            else:
                raise TypeError(f'{t} not supported {X[:,i]}')
        return kfs

class ProductKernel(BaseKernel):

    """
    Implements a product kernel given a list of kernel functions
    """
    
    def __init__(self) -> None:
        pass

    def base_kf(self, x) -> Any:
        prod = 1
        x = np.array(x).reshape((x.shape[0], -1))
        for feat in range(x.shape[1]):
            coeff = self.kernel_coefficients[feat]
            try:
                partial_res = self.kernel_functions[feat](x[:,feat]/coeff)/coeff
            except:
                raise IndexError(f'Out of bound: sample={str(x)}')
            prod*=partial_res
        return prod

    def fit(self, X: np.ndarray, kernel_functions: Optional[Iterable[Callable]] = None, kernel_coefficients: Optional[Union[Iterable[float], str]] = None):
        self.X = np.array(X)
        if kernel_functions is None:
            self.kernel_functions = self.infer_kernel_from_data(X)

        if isinstance(kernel_coefficients, str):
            if kernel_coefficients == 'silverman':
                self.kernel_coefficients = self.infer_kernel_coefficients_from_data(X)
            else:
                self.kernel_coefficients = None
        else:
            self.kernel_coefficients = self.infer_kernel_coefficients_from_data(X)
        kernel = lambda x: np.average(self.base_kf(x-self.X))
        kernel = np.vectorize(kernel)
        self._kernel = kernel
        self._kernel = gaussian_kde(self.X.reshape((-1,)), bw_method='silverman')

    def predict(self, predt: np.ndarray):
        print('predt is ', predt)
        print('ret is ', self._kernel(predt))
        return self._kernel(predt)
    
    def gradient(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the gradient squared error for estimated probabilities.'''
        y = dtrain.get_label()
        ret = self.predict(predt).reshape((-1,))-y
        print('y is ', y, 'ret is', ret)
        return ret
    
    def hessian(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the gradient squared error for estimated probabilities.'''
        return np.ones_like(dtrain.get_label())

    def __call__(self, y_pred, y_true) -> Any:
        return self.gradient(y_pred, y_true), self.hessian(y_pred, y_true)


class GraphKernel(BaseKernel):

    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, causal_graph: 'nx.Digraph'):
        self.X = X
        self.kernel_functions = self.infer_kernel_from_data(X)
        self.kernel_coefficients = self.infer_kernel_coefficients_from_data(X)
        self._causal_graph = copy(causal_graph)
        self.feature_list = list(nx.topological_sort(causal_graph))

        nx.set_node_attributes(self._causal_graph, {feat: lambda x: self.base_kf(x, feat, causal_graph) for feat in self.feature_list}, name='kernel')
        
        self._causal_graph = causal_graph
    
    def base_kf(self, x, feature: int, causal_graph: 'nx.Digraph') -> Any:
        preceding_features = causal_graph.predecessors(feature)
        prod = 1
        x = np.array(x)
        for feat in preceding_features:
            coeff = self.kernel_coefficients[feat]
            try:
                """print('Kernel funcs', len(self.kernel_functions))
                print('x is ', x)   
                print('feat is ', feat)
                print('preceding features are ', list(preceding_features))"""
                partial_res = self.kernel_functions[feat](x[feat]/coeff)/coeff
            except:
                raise IndexError(f'Out of bound: current feature={feature}, current feature in loop={str(feat)}, sample={str(x)}, preceding features={str(list(preceding_features))}, origin feature={feature}, feature topo order = {list(nx.topological_sort(causal_graph))}, feature preceding 2 = {list(causal_graph.predecessors(2))}')
            prod*=partial_res
        return prod

    def predict(self, predt: np.ndarray, feat: Hashable):
        return self._causal_graph.nodes[feat]['kernel'](predt)
    
    def __call__(self, predt: np.ndarray, feat: Hashable):
        return self._causal_graph.nodes[feat]['kernel'](predt)

def map_indeces_to_sample(indeces: Iterable, feature_list: Iterable, data: Iterable) -> 'np.Array':
    try:
        data = np.array(data)
    except:
        TypeError(f'Could not cast data of type {type(data)} to numpy array')
    try:
        feature_list = np.array(feature_list)
    except:
        TypeError(f'Could not cast feature_list of type {type(feature_list)} to numpy array')
    try:
        indeces = np.array(list(indeces))
    except:
        TypeError(f'Could not cast indeces of type {type(indeces)} to numpy array')
    if len(indeces.shape)==1:
        indeces = np.reshape(indeces, (1,)+indeces.shape)
    return np.array([[data[sample_index, feature_list[i]] for i, sample_index in enumerate(indeces_of_one_sample)]for indeces_of_one_sample in indeces]) 

def train_domain_shifter(d1: 'pd.DataFrame', d2: 'pd.DataFrame'):
    regressors = []
    types = np.array(d2.dtypes)
    d1 = copy(d1)
    for i, column in enumerate(d2.columns):
        if not pd.api.types.is_integer_dtype(types[i]) and not pd.api.types.is_float_dtype(types[i]):
            d1 = d1.drop(column, axis=1)
            continue
        pk = ProductKernel()
        print(f'Fitting {column}')
        try:
            pk.fit(pd.DataFrame(d2[column]))
        except:
            print(f'Could not fit {column}')
            continue
        pk_d1 = ProductKernel()
        pk_d1.fit(pd.DataFrame(d1[column]))
        y_train = np.array(d2[column].sample(len(d1), replace=True))
        # y_train = pk.predict(y_train)
        dtrain = xgb.DMatrix(d1, y_train)
        """reg = xgb.train({'tree_method': 'hist', 'seed': 1994, 'eta': 1e-1},  # any other tree method is fine.
           dtrain=dtrain,
           num_boost_round=10,
           obj=pk)"""
        reg = xgb.train({'tree_method': 'hist', 'seed': 1994, 'eta': 1e-1, 'objective': 'reg:squarederror'},  # any other tree method is fine.
           dtrain=dtrain,
           num_boost_round=10)
        regressors.append(reg)
    return regressors

def shift_domain(df: 'pd.DataFrame', regressors: Iterable):
    df_shifted = copy(df)
    for i, reg in enumerate(regressors):
        df_shifted.iloc[:,i] = reg.predict(xgb.DMatrix(df))
    return df_shifted

def check_sample_is_original(indeces: Iterable):
    return len(np.unique(indeces)) == 1

def causal_augmentation(data: 'pd.DataFrame', causal_graph: nx.Graph, kernel: GraphKernel, threshold: float, aug_coeff: float, sample_size: int = 10) -> Tuple['np.array', 'np.array']:

    weight_dict = dict()
    weight_dict[tuple()] = 1
    feature_list = list(nx.topological_sort(causal_graph))
    data = np.array(data.sample(sample_size))
    i = -1

    for feature in tqdm(feature_list):
        i+=1
        new_dict = dict()
        for leaf in tqdm(weight_dict.keys()):
            kernel_results = []
            augmented_sample = np.zeros_like(feature_list)
            for i in range(len(leaf)):
                augmented_sample[feature_list[i]] = data[leaf[i], feature_list[i]]
            for k in range(len(data)):
                truncated_data = np.zeros_like(feature_list)
                for i in range(len(leaf)):
                    truncated_data[feature_list[i]] = data[k, feature_list[i]]
                try: #senon riesci ti voglio bene lo stesso
                    # kernel_results.append(kernel(augmented_sample-truncated_data, feature))
                    kernel_results.append(kernel(augmented_sample-truncated_data, feature))
                except:
                    raise ValueError(f'Shapes {augmented_sample.shape}, {truncated_data.shape} not compatible\
                                        \nSamples are {augmented_sample}, {truncated_data}')
            den = sum(kernel_results)
            for current_datum_index in range(len(data)):
                new_leaf = leaf + (current_datum_index,)
                num = kernel_results[current_datum_index]
                if den>0:
                    w = num/den
                else:
                    w = 0
                w = weight_dict[leaf]*w
                if w>threshold/len(data):
                    new_dict[new_leaf] = w
                
        weight_dict = new_dict
    
    augmented_data_set = map_indeces_to_sample(weight_dict.keys(), feature_list, data)
    augmented_data_set = np.vstack((data, augmented_data_set))

    weights_aug = aug_coeff*len(data)*np.array(list(weight_dict.values()))
    weights_orig = (1-aug_coeff)*np.ones((len(data),))
    weights = np.hstack((weights_orig, weights_aug))

    return augmented_data_set, weights

def fast_causal_augmentation(data: 'pd.DataFrame', causal_graph: nx.Graph, kernel: GraphKernel, threshold: float, aug_coeff: float, sample_size: int = 10, topk: Optional[int] = None) -> Tuple['np.array', 'np.array']:

    weight_dict = dict()
    weight_dict[tuple()] = 1
    feature_list = list(nx.topological_sort(causal_graph))
    columns = None
    if isinstance(data, pd.DataFrame):
        columns = data.columns
    data = np.array(data.sample(sample_size))
    i = -1
    for feature in tqdm(feature_list):
        i+=1
        new_dict = dict()
        for leaf in weight_dict.keys():
            feature_values = list()
            kernel_results = []
            augmented_sample = np.zeros_like(feature_list)
            for i in range(len(leaf)):
                augmented_sample[feature_list[i]] = data[leaf[i], feature_list[i]]
            for k in range(len(data)):
                truncated_data = np.zeros_like(feature_list)
                for i in range(len(leaf)):
                    truncated_data[feature_list[i]] = data[k, feature_list[i]]
                try:
                    # kernel_results.append(kernel(augmented_sample-truncated_data, feature))
                    kernel_results.append(kernel(augmented_sample-truncated_data, feature))
                except:
                    raise ValueError(f'Shapes {augmented_sample.shape}, {truncated_data.shape} not compatible\
                                        \nSamples are {augmented_sample}, {truncated_data}')
            den = sum(kernel_results)
            for current_datum_index in range(len(data)):
                if data[current_datum_index, feature] in feature_values:
                    continue
                else:
                    feature_values.append(data[current_datum_index, feature])
                new_leaf = leaf + (current_datum_index,)
                num = kernel_results[current_datum_index]
                if den>0:
                    w = num/den
                else:
                    w = 0
                w = weight_dict[leaf]*w
                if w>threshold/len(data):
                    new_dict[new_leaf] = w
            if topk is not None:
                new_dict = dict(sorted(new_dict.items(), key=lambda x: x[1], reverse=True)[:topk])
                
        weight_dict = new_dict
    
    if len(weight_dict.keys())==0:
        augmented_data_set = np.empty_like(data)
    augmented_data_set = map_indeces_to_sample(weight_dict.keys(), feature_list, data)
    augmented_data_set = np.vstack((data, augmented_data_set))

    weights_aug = aug_coeff*len(data)*np.array(list(weight_dict.values()))
    weights_orig = (1-aug_coeff)*np.ones((len(data),))
    weights = np.hstack((weights_orig, weights_aug))

    if columns is not None:
        return pd.DataFrame(augmented_data_set, columns=columns), weights

    return augmented_data_set, weights


def get_default_kernel_functions_from_df(df: 'pd.DataFrame') -> List[Callable]:
    types = df.dtypes
    kfs = []
    for t in types:
        if pd.api.types.is_integer_dtype(t):
            kfs.append(lambda x: 1 if x==0 else 0)
        elif pd.api.types.is_float_dtype(t):
            kfs.append(lambda x: np.power(2*np.pi, -0.5)*np.exp(-np.power(x, 2)/2))
        else:
            raise TypeError(f'{t} not supported')
    return kfs

def causal_data_augmentation(d1: 'pd.DataFrame', d2: 'pd.DataFrame', causal_graph: nx.Graph, shift_variable: str, n: Optional[int] = None):
    
    feature_list = list(nx.topological_sort(causal_graph))

    dependent_variables = []
    independent_variables = []

    i_shift = 0
    for c in d2.columns:
        if c == shift_variable:
            break
        else:
            i_shift+=1

    for node in causal_graph.nodes:
        if node == shift_variable:
            continue
        if nx.has_path(causal_graph, i_shift, node):
            dependent_variables.append(node)
        else:
            independent_variables.append(node)

    source_data = np.array(d1)
    target_data = np.array(d2)
    if n is None:
        data_aug = np.empty([source_data.shape[0], 0])
    else:
        data_aug = np.empty([n, 0])

    features_so_far = {}

    for i, feature in tqdm(enumerate(feature_list)):
        features_so_far[feature] = i
        new_values = []
        predecessors = list(causal_graph.predecessors(feature)) + [feature]
        if feature in independent_variables:
            # sample from source domain
            target_column = source_data[:,feature]
            # domain = source_data[:,feature_list[:i+1]]
            domain = source_data[:,predecessors]
        else:
            # sample on target domain by conditioning
            target_column = target_data[:,feature]
            domain = target_data[:,predecessors]
        try:
            dim_reduction = False
            kernel = gaussian_kde(domain.T)
        except:
            dim_reduction = True
            pca = PCA(n_components=0.99)
            domain = pca.fit_transform(domain)
            kernel = gaussian_kde(domain.T)
        for sample in data_aug[:,[features_so_far[p] for p in predecessors if p!=feature]]:
            repeated_samples = np.tile(sample,(len(target_column),1))
            new_samples = np.hstack((repeated_samples, target_column.reshape((-1,1))))
            if dim_reduction:
                new_samples = pca.transform(new_samples)
            try:
                scores = kernel.pdf(new_samples.T)
            except:
                raise TypeError(f'Type of new_samples is {new_samples}, found types in new_samples {set([type(x) for x in new_samples.flatten()])}')
            weights = scores/sum(scores)
            if np.isnan(weights).any() or sum(scores)<=0:
                # If there are nan values, replace them with a uniform distribution
                weights = np.where(np.isnan(weights), 0, weights)
                # Normalize the weights so they sum to 1
                if np.sum(weights) <= 0:
                    weights = np.ones_like(weights)/len(weights)
                else:
                    weights = weights / np.sum(weights)
            weighted_mean = np.average(target_column, weights=weights)
            weighted_std = np.sqrt(np.average((target_column - weighted_mean)**2, weights=weights))
            # Calculate the interquartile range
            iqr_value = iqr(target_column)
            # Calculate the Silverman bandwidth
            silverman_bandwidth = 0.9 * min(weighted_std, iqr_value / 1.34) * len(target_column) ** (-1/5)
            
            new_value = np.random.choice(target_column, p=weights)+np.random.normal(scale=silverman_bandwidth)
            new_values.append(new_value)
        data_aug = np.hstack((data_aug, np.reshape(new_values, (-1,1))))
    
    if isinstance(d1, pd.DataFrame):
        return pd.DataFrame(data_aug, columns=d1.columns[feature_list])

    return data_aug