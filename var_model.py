"""
Vector Autoregression (VAR) Model for Factor Forecasting
Based on Chincarini's approach to multi-factor asset pricing and forecasting
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import config

warnings.filterwarnings('ignore')

class ChincariniVARModel:
    def __init__(self, max_lags: int = None):
        self.max_lags = max_lags or config.VAR_SETTINGS['max_lags']
        self.trend = config.VAR_SETTINGS['trend']
        self.method = config.VAR_SETTINGS['method']
        
        self.model = None
        self.fitted_model = None
        self.factor_data = None
        self.factor_names = None
        self.optimal_lags = None
        self.residuals = None
        self.forecast_results = None
        
    def prepare_factor_data(self, technical_factors: pd.DataFrame, 
                          fundamental_factors: pd.DataFrame,
                          sentiment_factors: pd.DataFrame,
                          economic_indicators: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare and combine all factor data for VAR modeling
        Following Chincarini's multi-factor approach
        """
        print("Preparing factor data for VAR model...")
        
        # Combine all factors
        factor_list = []
        
        if not technical_factors.empty:
            # Use the aggregated technical factors (PCA components)
            factor_list.append(technical_factors)
            print(f"Added {technical_factors.shape[1]} technical factors")
        
        if not fundamental_factors.empty:
            # Use the aggregated fundamental factors
            factor_list.append(fundamental_factors)
            print(f"Added {fundamental_factors.shape[1]} fundamental factors")
        
        if not sentiment_factors.empty:
            # Use the aggregated sentiment factors
            factor_list.append(sentiment_factors)
            print(f"Added {sentiment_factors.shape[1]} sentiment factors")
        
        if economic_indicators is not None and not economic_indicators.empty:
            # Add key economic indicators
            econ_factors = self._process_economic_indicators(economic_indicators)
            if not econ_factors.empty:
                factor_list.append(econ_factors)
                print(f"Added {econ_factors.shape[1]} economic factors")
        
        if not factor_list:
            raise ValueError("No valid factor data provided")
        
        # Combine all factors
        combined_factors = pd.concat(factor_list, axis=1)
        
        # Handle missing values
        combined_factors = combined_factors.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN rows
        combined_factors = combined_factors.dropna()
        
        print(f"Final factor matrix shape: {combined_factors.shape}")
        
        self.factor_data = combined_factors
        self.factor_names = list(combined_factors.columns)
        
        return combined_factors
    
    def _process_economic_indicators(self, economic_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Process economic indicators into usable factors
        """
        econ_factors = pd.DataFrame()
        
        try:
            # Interest rate factors
            if 'fed_rate' in economic_data and not economic_data['fed_rate'].empty:
                fed_rate = economic_data['fed_rate'].iloc[:, 0]
                econ_factors['Fed_Rate'] = fed_rate
                econ_factors['Fed_Rate_Change'] = fed_rate.diff()
            
            # Yield curve
            if '10y_treasury' in economic_data and '2y_treasury' in economic_data:
                if not economic_data['10y_treasury'].empty and not economic_data['2y_treasury'].empty:
                    y10 = economic_data['10y_treasury'].iloc[:, 0]
                    y2 = economic_data['2y_treasury'].iloc[:, 0]
                    econ_factors['Yield_Curve_Slope'] = y10 - y2
                    econ_factors['Yield_10Y'] = y10
            
            # Volatility (VIX)
            if 'vix' in economic_data and not economic_data['vix'].empty:
                vix = economic_data['vix'].iloc[:, 0]
                econ_factors['VIX'] = vix
                econ_factors['VIX_Change'] = vix.diff()
                econ_factors['VIX_Spike'] = (vix > vix.rolling(window=252).quantile(0.9)).astype(int)
            
            # GDP Growth
            if 'gdp_growth' in economic_data and not economic_data['gdp_growth'].empty:
                gdp = economic_data['gdp_growth'].iloc[:, 0]
                econ_factors['GDP_Growth'] = gdp.pct_change()
            
            # Unemployment
            if 'unemployment' in economic_data and not economic_data['unemployment'].empty:
                unemp = economic_data['unemployment'].iloc[:, 0]
                econ_factors['Unemployment'] = unemp
                econ_factors['Unemployment_Change'] = unemp.diff()
                
        except Exception as e:
            print(f"Error processing economic indicators: {e}")
        
        return econ_factors.fillna(method='ffill').dropna()
    
    def test_stationarity(self, alpha: float = 0.05) -> Dict[str, Dict]:
        """
        Test stationarity of all factors using Augmented Dickey-Fuller test
        """
        if self.factor_data is None:
            raise ValueError("Factor data not prepared. Call prepare_factor_data() first.")
        
        stationarity_results = {}
        
        print("Testing stationarity of factors...")
        
        for column in self.factor_data.columns:
            try:
                series = self.factor_data[column].dropna()
                
                # Perform ADF test
                adf_result = adfuller(series, autolag='AIC')
                
                stationarity_results[column] = {
                    'adf_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < alpha
                }
                
                status = "Stationary" if adf_result[1] < alpha else "Non-stationary"
                print(f"{column}: {status} (p-value: {adf_result[1]:.4f})")
                
            except Exception as e:
                print(f"Error testing stationarity for {column}: {e}")
                stationarity_results[column] = {
                    'error': str(e),
                    'is_stationary': False
                }
        
        return stationarity_results
    
    def make_stationary(self, method: str = 'diff') -> pd.DataFrame:
        """
        Make non-stationary series stationary through differencing or detrending
        """
        if self.factor_data is None:
            raise ValueError("Factor data not prepared.")
        
        stationary_data = self.factor_data.copy()
        
        # Test stationarity first
        stationarity_results = self.test_stationarity()
        
        for column, result in stationarity_results.items():
            if not result.get('is_stationary', False):
                print(f"Making {column} stationary using {method}...")
                
                if method == 'diff':
                    # First difference
                    stationary_data[column] = stationary_data[column].diff()
                elif method == 'log_diff':
                    # Log difference (for positive series)
                    if (stationary_data[column] > 0).all():
                        stationary_data[column] = np.log(stationary_data[column]).diff()
                    else:
                        # Fall back to regular differencing
                        stationary_data[column] = stationary_data[column].diff()
                elif method == 'detrend':
                    # Linear detrending
                    from scipy import signal
                    stationary_data[column] = signal.detrend(stationary_data[column])
        
        # Remove NaN values created by differencing
        stationary_data = stationary_data.dropna()
        
        print(f"Stationary data shape: {stationary_data.shape}")
        
        self.factor_data = stationary_data
        return stationary_data
    
    def select_optimal_lags(self) -> int:
        """
        Select optimal lag length using information criteria
        """
        if self.factor_data is None:
            raise ValueError("Factor data not prepared.")
        
        print("Selecting optimal lag length...")
        
        # Initialize VAR model
        model = VAR(self.factor_data)
        
        # Select optimal lag using multiple criteria
        lag_selection = model.select_order(maxlags=self.max_lags)
        
        print("Lag selection criteria:")
        print(f"AIC: {lag_selection.aic}")
        print(f"BIC: {lag_selection.bic}")
        print(f"FPE: {lag_selection.fpe}")
        print(f"HQIC: {lag_selection.hqic}")
        
        # Use AIC as default criterion (can be changed)
        optimal_lags = lag_selection.aic
        
        print(f"Selected optimal lags: {optimal_lags}")
        
        self.optimal_lags = optimal_lags
        return optimal_lags
    
    def fit_var_model(self, lags: int = None) -> None:
        """
        Fit the VAR model to the factor data
        """
        if self.factor_data is None:
            raise ValueError("Factor data not prepared.")
        
        if lags is None:
            if self.optimal_lags is None:
                lags = self.select_optimal_lags()
            else:
                lags = self.optimal_lags
        
        print(f"Fitting VAR model with {lags} lags...")
        
        # Initialize and fit VAR model
        self.model = VAR(self.factor_data)
        self.fitted_model = self.model.fit(lags, trend=self.trend, method=self.method)
        
        print("VAR model fitted successfully!")
        print(f"Model summary:")
        print(f"- Endogenous variables: {len(self.factor_names)}")
        print(f"- Lags: {lags}")
        print(f"- Observations: {self.fitted_model.nobs}")
        print(f"- Log-likelihood: {self.fitted_model.llf:.2f}")
        print(f"- AIC: {self.fitted_model.aic:.2f}")
        print(f"- BIC: {self.fitted_model.bic:.2f}")
        
        # Store residuals for diagnostics
        self.residuals = self.fitted_model.resid
        
    def model_diagnostics(self) -> Dict:
        """
        Perform comprehensive model diagnostics
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit_var_model() first.")
        
        diagnostics = {}
        
        print("Performing model diagnostics...")
        
        # 1. Residual autocorrelation test (Ljung-Box)
        print("Testing residual autocorrelation...")
        ljung_box_results = {}
        
        for i, column in enumerate(self.factor_names):
            try:
                lb_result = acorr_ljungbox(self.residuals.iloc[:, i], lags=10, return_df=True)
                ljung_box_results[column] = {
                    'statistic': lb_result['lb_stat'].iloc[-1],
                    'p_value': lb_result['lb_pvalue'].iloc[-1],
                    'no_autocorr': lb_result['lb_pvalue'].iloc[-1] > 0.05
                }
            except Exception as e:
                print(f"Error in Ljung-Box test for {column}: {e}")
        
        diagnostics['ljung_box'] = ljung_box_results
        
        # 2. Normality of residuals (Jarque-Bera test)
        try:
            from statsmodels.stats.diagnostic import jarque_bera
            jb_results = {}
            
            for i, column in enumerate(self.factor_names):
                jb_stat, jb_pvalue, _, _ = jarque_bera(self.residuals.iloc[:, i])
                jb_results[column] = {
                    'statistic': jb_stat,
                    'p_value': jb_pvalue,
                    'is_normal': jb_pvalue > 0.05
                }
            
            diagnostics['jarque_bera'] = jb_results
        except Exception as e:
            print(f"Error in Jarque-Bera test: {e}")
        
        # 3. Model stability
        try:
            # Check if characteristic roots are inside unit circle
            roots = self.fitted_model.roots
            max_root = np.max(np.abs(roots))
            diagnostics['stability'] = {
                'max_root': max_root,
                'is_stable': max_root < 1.0
            }
            
            if max_root < 1.0:
                print("Model is stable (all roots inside unit circle)")
            else:
                print("Warning: Model may be unstable (some roots outside unit circle)")
                
        except Exception as e:
            print(f"Error checking model stability: {e}")
        
        return diagnostics
    
    def forecast_factors(self, steps: int = 22) -> pd.DataFrame:
        """
        Generate multi-step ahead forecasts for all factors
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit_var_model() first.")
        
        print(f"Generating {steps}-step ahead forecasts...")
        
        # Generate forecasts
        forecast_result = self.fitted_model.forecast(self.factor_data.values, steps)
        
        # Create forecast DataFrame
        forecast_dates = pd.date_range(
            start=self.factor_data.index[-1] + pd.Timedelta(days=1),
            periods=steps,
            freq='B'  # Business days
        )
        
        forecast_df = pd.DataFrame(
            forecast_result,
            index=forecast_dates,
            columns=self.factor_names
        )
        
        # Also get forecast error variance (confidence intervals)
        try:
            forecast_errors = self.fitted_model.forecast_cov(steps)
            
            # Calculate confidence intervals (95%)
            confidence_intervals = {}
            
            for i, factor in enumerate(self.factor_names):
                std_errors = np.sqrt(np.diagonal(forecast_errors)[:, i])
                
                confidence_intervals[f'{factor}_lower'] = forecast_df[factor] - 1.96 * std_errors
                confidence_intervals[f'{factor}_upper'] = forecast_df[factor] + 1.96 * std_errors
            
            # Add confidence intervals to forecast DataFrame
            for key, values in confidence_intervals.items():
                forecast_df[key] = values
                
        except Exception as e:
            print(f"Error calculating confidence intervals: {e}")
        
        self.forecast_results = forecast_df
        
        print("Factor forecasts generated successfully!")
        return forecast_df
    
    def analyze_factor_interactions(self) -> Dict:
        """
        Analyze interactions between factors using impulse response functions
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit_var_model() first.")
        
        print("Analyzing factor interactions...")
        
        interactions = {}
        
        try:
            # Impulse Response Functions
            irf = self.fitted_model.irf(periods=20)
            
            # Store IRF data
            interactions['impulse_responses'] = {
                'values': irf.irfs,
                'factor_names': self.factor_names
            }
            
            # Forecast Error Variance Decomposition
            fevd = self.fitted_model.fevd(periods=20)
            
            interactions['variance_decomposition'] = {
                'values': fevd.decomp,
                'factor_names': self.factor_names
            }
            
            print("Factor interaction analysis completed!")
            
        except Exception as e:
            print(f"Error in factor interaction analysis: {e}")
            
        return interactions
    
    def get_factor_loadings(self) -> pd.DataFrame:
        """
        Extract factor loadings (coefficients) from the fitted VAR model
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit_var_model() first.")
        
        # Get coefficient matrix
        coef_matrix = self.fitted_model.coefs
        
        # Reshape and create DataFrame
        loadings_list = []
        
        for lag in range(coef_matrix.shape[0]):
            lag_df = pd.DataFrame(
                coef_matrix[lag],
                index=self.factor_names,
                columns=[f'{name}_lag_{lag+1}' for name in self.factor_names]
            )
            loadings_list.append(lag_df)
        
        # Combine all lags
        factor_loadings = pd.concat(loadings_list, axis=1)
        
        return factor_loadings
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted VAR model
        """
        import pickle
        
        model_data = {
            'fitted_model': self.fitted_model,
            'factor_data': self.factor_data,
            'factor_names': self.factor_names,
            'optimal_lags': self.optimal_lags,
            'forecast_results': self.forecast_results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a previously saved VAR model
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.fitted_model = model_data['fitted_model']
        self.factor_data = model_data['factor_data']
        self.factor_names = model_data['factor_names']
        self.optimal_lags = model_data['optimal_lags']
        self.forecast_results = model_data.get('forecast_results')
        
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    print("Chincarini VAR Model module ready!")
    print("Use with factor data to create multi-factor forecasting models.") 