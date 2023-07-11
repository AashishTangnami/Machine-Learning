import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass


FEATURES = ["height", "weight", "bark_days", "ear_head_ratio"]

@dataclass
class params_gaussian:
    mu: float
    sigma: float
        
    def __repr__(self):
        return f"params_gaussian(mu={self.mu:.3f}, sigma={self.sigma:.3f})"


@dataclass
class params_binomial:
    n: int
    p: float
        
    def __repr__(self):
        return f"params_binomial(n={self.n:.3f}, p={self.p:.3f})"


@dataclass
class params_uniform:
    a: int
    b: int
        
    def __repr__(self):
        return f"params_uniform(a={self.a:.3f}, b={self.b:.3f})"
    
    
breed_params = {
    0: {
        "height": params_gaussian(mu=35, sigma=1.5),
        "weight": params_gaussian(mu=20, sigma=1),
        "bark_days": params_binomial(n=30, p=0.8),
        "ear_head_ratio": params_uniform(a=0.6, b=0.1)
    },
    
    1: {
        "height": params_gaussian(mu=30, sigma=2),
        "weight": params_gaussian(mu=25, sigma=5),
        "bark_days": params_binomial(n=30, p=0.5),
        "ear_head_ratio": params_uniform(a=0.2, b=0.5)
    },
    
    2: {
        "height": params_gaussian(mu=40, sigma=3.5),
        "weight": params_gaussian(mu=32, sigma=3),
        "bark_days": params_binomial(n=30, p=0.3),
        "ear_head_ratio": params_uniform(a=0.1, b=0.3)
    }
    
}

def round_dict(nested_dict):
    rounded_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            rounded_dict[key] = round_dict(value)
        else:
            rounded_dict[key] = round(value, 3)
    return rounded_dict


def generate_data_for_breed(breed, features, n_samples, params, gg, bg, ug):
    """
    Generate synthetic data for a specific breed of dogs based on given features and parameters.

    Parameters:
        - breed (str): The breed of the dog for which data is generated.
        - features (list[str]): List of features to generate data for (e.g., "height", "weight", "bark_days", "ear_head_ratio").
        - n_samples (int): Number of samples to generate for each feature.
        - params (dict): Dictionary containing parameters for each breed and its features.

    Returns:
        - df (pandas.DataFrame): A DataFrame containing the generated synthetic data.
            The DataFrame will have columns for each feature and an additional column for the breed.
    """
    
    df = pd.DataFrame()
    
    for feature in features:
        match feature:
            case "height" | "weight":
                df[feature] = gg(params[breed][feature].mu, params[breed][feature].sigma, n_samples)
                
            case "bark_days":
                df[feature] = bg(params[breed][feature].n, params[breed][feature].p, n_samples)
                                       
            case "ear_head_ratio":
                df[feature] = ug(params[breed][feature].a, params[breed][feature].b, n_samples)    
    
    df["breed"] = breed
    
    return df


def generate_data(gaussian_generator, binomial_generator, uniform_generator):
    # Generate data for each breed
    df_0 = generate_data_for_breed(breed=0, features=FEATURES, n_samples=1200, params=breed_params, gg=gaussian_generator, bg=binomial_generator, ug=uniform_generator)
    df_1 = generate_data_for_breed(breed=1, features=FEATURES, n_samples=1350, params=breed_params, gg=gaussian_generator, bg=binomial_generator, ug=uniform_generator)
    df_2 = generate_data_for_breed(breed=2, features=FEATURES, n_samples=900, params=breed_params, gg=gaussian_generator, bg=binomial_generator, ug=uniform_generator)

    # Concatenate all breeds into a single dataframe
    df_all_breeds = pd.concat([df_0, df_1, df_2]).reset_index(drop=True)

    # Shuffle the data
    df_all_breeds = df_all_breeds.sample(frac = 1)
    
    return df_all_breeds



def compute_training_params(df, features):
    """
    Computes the estimated parameters for training a model based on the provided dataframe and features.

    Args:
        df (pandas.DataFrame): The dataframe containing the training data.
        features (list): A list of feature names to consider.

    Returns:
        - params_dict (dict): A dictionary that contains the estimated parameters for each breed and feature.  
    """
    
    # Dict that should contain the estimated parameters
    params_dict = {}
    
    
    ### START CODE HERE ###
    
    # Loop over the breeds
    for breed in range(3): # @REPLACE for None in None:
        
        # Slice the original df to only include data for the current breed and the feature columns
        # For reference in slicing with pandas, you can use the df_breed.groupby function followed by .get_group
        # or you can use the syntax df[df['breed'] == group]
        df_breed = df[df["breed"] == breed][features] # @REPLACE df_breed = df[df["breed"] == None][features]
        
        # Initialize the inner dict
        inner_dict = {}
        
        # Loop over the columns of the sliced dataframe
        # You can get the columns of a dataframe like this: dataframe.columns
        for col in df_breed.columns: 
            match col: 
                case "height" | "weight":
                    mu, sigma = estimate_gaussian_params(df_breed[col])
                    m = {"mu":mu, "sigma":sigma}
                    
                case "bark_days":
                    n, p = estimate_binomial_params(df_breed[col])
                    m = {"n":n, "p":p}

                case "ear_head_ratio":
                    a, b = estimate_uniform_params(df_breed[col])
                    m = {"a":a, "b":b}
            
            # Save the dataclass object within the inner dict
            inner_dict[col] = m
        
        # Save inner dict within outer dict
        params_dict[breed] = inner_dict
    
    ### END CODE HERE ###

    return params_dict


def estimate_gaussian_params(sample):
    ### START CODE HERE ###
    mu = np.mean(sample)
    sigma = np.std(sample)
    ### END CODE HERE ###

    return mu, sigma


def estimate_binomial_params(sample):
    ### START CODE HERE ###
    n = 30
    p = (sample / n).mean()
    ### END CODE HERE ###

    return n, p


def estimate_uniform_params(sample):
    ### START CODE HERE ###
    a = sample.min()
    b = sample.max()
    ### END CODE HERE ###

    return a, b


def plot_gaussian_distributions(gaussian_0, gaussian_1, gaussian_2):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(gaussian_0, alpha=0.5, label="gaussian_0", bins=32)
    ax.hist(gaussian_1, alpha=0.5, label="gaussian_1", bins=32)
    ax.hist(gaussian_2, alpha=0.5, label="gaussian_2", bins=32)
    ax.set_title("Histograms of Gaussian distributions")
    ax.set_xlabel("Values")
    ax.set_ylabel("Frequencies")
    ax.legend()
    plt.show()
    
    
def plot_binomial_distributions(binomial_0, binomial_1, binomial_2):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(binomial_0, alpha=0.5, label="binomial_0")
    ax.hist(binomial_1, alpha=0.5, label="binomial_1")
    ax.hist(binomial_2, alpha=0.5, label="binomial_2")
    ax.set_title("Histograms of Binomial distributions")
    ax.set_xlabel("Values")
    ax.set_ylabel("Frequencies")
    ax.legend()
    plt.show()