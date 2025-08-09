"""
Fundamental Factor Extraction Module
Processes financial statement data and creates fundamental factors
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class FundamentalFactorExtractor:
    def __init__(self):
        self.fundamental_metrics = {}
        
    def process_income_statement(self, income_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract key metrics from income statement data
        """
        if income_data.empty:
            return pd.DataFrame()
            
        metrics = pd.DataFrame()
        
        try:
            # Revenue metrics
            if 'totalRevenue' in income_data.columns:
                metrics['Revenue'] = pd.to_numeric(income_data['totalRevenue'], errors='coerce')
                metrics['Revenue_Growth'] = metrics['Revenue'].pct_change()
                metrics['Revenue_Growth_3Y'] = metrics['Revenue'].rolling(window=3).apply(
                    lambda x: (x.iloc[-1] / x.iloc[0]) ** (1/3) - 1 if len(x) == 3 and x.iloc[0] != 0 else np.nan
                )
            
            # Profitability metrics
            if 'grossProfit' in income_data.columns:
                metrics['Gross_Profit'] = pd.to_numeric(income_data['grossProfit'], errors='coerce')
                if 'Revenue' in metrics.columns:
                    metrics['Gross_Margin'] = metrics['Gross_Profit'] / metrics['Revenue']
            
            if 'operatingIncome' in income_data.columns:
                metrics['Operating_Income'] = pd.to_numeric(income_data['operatingIncome'], errors='coerce')
                if 'Revenue' in metrics.columns:
                    metrics['Operating_Margin'] = metrics['Operating_Income'] / metrics['Revenue']
            
            if 'netIncome' in income_data.columns:
                metrics['Net_Income'] = pd.to_numeric(income_data['netIncome'], errors='coerce')
                if 'Revenue' in metrics.columns:
                    metrics['Net_Margin'] = metrics['Net_Income'] / metrics['Revenue']
                metrics['Net_Income_Growth'] = metrics['Net_Income'].pct_change()
            
            # Efficiency metrics
            if 'totalOperatingExpenses' in income_data.columns and 'Revenue' in metrics.columns:
                operating_expenses = pd.to_numeric(income_data['totalOperatingExpenses'], errors='coerce')
                metrics['Operating_Efficiency'] = operating_expenses / metrics['Revenue']
            
            # EBITDA approximation
            if all(col in income_data.columns for col in ['netIncome', 'incomeTaxExpense', 'interestExpense']):
                net_income = pd.to_numeric(income_data['netIncome'], errors='coerce')
                tax_expense = pd.to_numeric(income_data['incomeTaxExpense'], errors='coerce')
                interest_expense = pd.to_numeric(income_data['interestExpense'], errors='coerce')
                # Note: This is a simplified EBITDA calculation
                metrics['EBITDA_Approx'] = net_income + tax_expense + interest_expense
                if 'Revenue' in metrics.columns:
                    metrics['EBITDA_Margin'] = metrics['EBITDA_Approx'] / metrics['Revenue']
                    
        except Exception as e:
            print(f"Error processing income statement: {e}")
            
        return metrics.dropna(how='all')
    
    def process_balance_sheet(self, balance_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract key metrics from balance sheet data
        """
        if balance_data.empty:
            return pd.DataFrame()
            
        metrics = pd.DataFrame()
        
        try:
            # Asset metrics
            if 'totalAssets' in balance_data.columns:
                metrics['Total_Assets'] = pd.to_numeric(balance_data['totalAssets'], errors='coerce')
                metrics['Asset_Growth'] = metrics['Total_Assets'].pct_change()
            
            if 'totalCurrentAssets' in balance_data.columns:
                metrics['Current_Assets'] = pd.to_numeric(balance_data['totalCurrentAssets'], errors='coerce')
            
            if 'cashAndCashEquivalentsAtCarryingValue' in balance_data.columns:
                metrics['Cash'] = pd.to_numeric(balance_data['cashAndCashEquivalentsAtCarryingValue'], errors='coerce')
                if 'Total_Assets' in metrics.columns:
                    metrics['Cash_Ratio'] = metrics['Cash'] / metrics['Total_Assets']
            
            # Liability metrics
            if 'totalLiabilities' in balance_data.columns:
                metrics['Total_Liabilities'] = pd.to_numeric(balance_data['totalLiabilities'], errors='coerce')
            
            if 'totalCurrentLiabilities' in balance_data.columns:
                metrics['Current_Liabilities'] = pd.to_numeric(balance_data['totalCurrentLiabilities'], errors='coerce')
            
            # Equity metrics
            if 'totalStockholdersEquity' in balance_data.columns:
                metrics['Shareholders_Equity'] = pd.to_numeric(balance_data['totalStockholdersEquity'], errors='coerce')
                metrics['Equity_Growth'] = metrics['Shareholders_Equity'].pct_change()
            
            # Financial ratios
            if 'Current_Assets' in metrics.columns and 'Current_Liabilities' in metrics.columns:
                metrics['Current_Ratio'] = metrics['Current_Assets'] / metrics['Current_Liabilities']
            
            if 'Total_Liabilities' in metrics.columns and 'Shareholders_Equity' in metrics.columns:
                metrics['Debt_to_Equity'] = metrics['Total_Liabilities'] / metrics['Shareholders_Equity']
            
            if 'Total_Assets' in metrics.columns and 'Total_Liabilities' in metrics.columns:
                metrics['Asset_to_Liability'] = metrics['Total_Assets'] / metrics['Total_Liabilities']
                
        except Exception as e:
            print(f"Error processing balance sheet: {e}")
            
        return metrics.dropna(how='all')
    
    def process_cash_flow(self, cashflow_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract key metrics from cash flow statement
        """
        if cashflow_data.empty:
            return pd.DataFrame()
            
        metrics = pd.DataFrame()
        
        try:
            # Operating cash flow
            if 'operatingCashflow' in cashflow_data.columns:
                metrics['Operating_Cash_Flow'] = pd.to_numeric(cashflow_data['operatingCashflow'], errors='coerce')
                metrics['OCF_Growth'] = metrics['Operating_Cash_Flow'].pct_change()
            
            # Free cash flow approximation
            if all(col in cashflow_data.columns for col in ['operatingCashflow', 'capitalExpenditures']):
                ocf = pd.to_numeric(cashflow_data['operatingCashflow'], errors='coerce')
                capex = pd.to_numeric(cashflow_data['capitalExpenditures'], errors='coerce')
                metrics['Free_Cash_Flow'] = ocf - capex
                metrics['FCF_Growth'] = metrics['Free_Cash_Flow'].pct_change()
            
            # Investment metrics
            if 'capitalExpenditures' in cashflow_data.columns:
                metrics['Capital_Expenditures'] = pd.to_numeric(cashflow_data['capitalExpenditures'], errors='coerce')
            
            # Financing metrics
            if 'dividendPayout' in cashflow_data.columns:
                metrics['Dividend_Payments'] = pd.to_numeric(cashflow_data['dividendPayout'], errors='coerce')
                
        except Exception as e:
            print(f"Error processing cash flow: {e}")
            
        return metrics.dropna(how='all')
    
    def process_company_overview(self, overview_data: pd.DataFrame) -> Dict:
        """
        Extract key metrics from company overview
        """
        if overview_data.empty:
            return {}
            
        overview_metrics = {}
        
        try:
            # Market metrics
            market_fields = [
                'MarketCapitalization', 'PERatio', 'PEGRatio', 'BookValue',
                'DividendPerShare', 'DividendYield', 'EPS', 'RevenuePerShareTTM',
                'ProfitMargin', 'OperatingMarginTTM', 'ReturnOnAssetsTTM',
                'ReturnOnEquityTTM', 'RevenueTTM', 'GrossProfitTTM', 'EBITDA',
                'DilutedEPSTTM', 'QuarterlyEarningsGrowthYOY', 'QuarterlyRevenueGrowthYOY',
                'AnalystTargetPrice', 'TrailingPE', 'ForwardPE', 'PriceToSalesRatioTTM',
                'PriceToBookRatio', 'EVToRevenue', 'EVToEBITDA', 'Beta',
                '52WeekHigh', '52WeekLow', '50DayMovingAverage', '200DayMovingAverage'
            ]
            
            for field in market_fields:
                if field in overview_data.columns and not overview_data[field].empty:
                    value = overview_data[field].iloc[0] if len(overview_data[field]) > 0 else None
                    if value and value != 'None':
                        try:
                            overview_metrics[field] = float(value)
                        except:
                            overview_metrics[field] = value
                            
        except Exception as e:
            print(f"Error processing company overview: {e}")
            
        return overview_metrics
    
    def calculate_financial_ratios(self, income_metrics: pd.DataFrame, 
                                 balance_metrics: pd.DataFrame,
                                 cashflow_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived financial ratios from multiple statements
        """
        ratios = pd.DataFrame()
        
        try:
            # ROE calculation
            if 'Net_Income' in income_metrics.columns and 'Shareholders_Equity' in balance_metrics.columns:
                ratios['ROE'] = income_metrics['Net_Income'] / balance_metrics['Shareholders_Equity']
            
            # ROA calculation
            if 'Net_Income' in income_metrics.columns and 'Total_Assets' in balance_metrics.columns:
                ratios['ROA'] = income_metrics['Net_Income'] / balance_metrics['Total_Assets']
            
            # Asset Turnover
            if 'Revenue' in income_metrics.columns and 'Total_Assets' in balance_metrics.columns:
                ratios['Asset_Turnover'] = income_metrics['Revenue'] / balance_metrics['Total_Assets']
            
            # Financial Leverage
            if 'Total_Assets' in balance_metrics.columns and 'Shareholders_Equity' in balance_metrics.columns:
                ratios['Financial_Leverage'] = balance_metrics['Total_Assets'] / balance_metrics['Shareholders_Equity']
            
            # Cash Flow to Debt ratio
            if 'Operating_Cash_Flow' in cashflow_metrics.columns and 'Total_Liabilities' in balance_metrics.columns:
                ratios['CF_to_Debt'] = cashflow_metrics['Operating_Cash_Flow'] / balance_metrics['Total_Liabilities']
            
            # Quality metrics
            if 'Operating_Cash_Flow' in cashflow_metrics.columns and 'Net_Income' in income_metrics.columns:
                ratios['CF_to_NI_Quality'] = cashflow_metrics['Operating_Cash_Flow'] / income_metrics['Net_Income']
                
        except Exception as e:
            print(f"Error calculating financial ratios: {e}")
            
        return ratios.dropna(how='all')
    
    def create_fundamental_factor(self, fundamental_data: Dict) -> pd.DataFrame:
        """
        Create comprehensive fundamental factor from all financial data
        """
        all_metrics = []
        
        # Process each component
        if 'income_statement' in fundamental_data:
            income_metrics = self.process_income_statement(fundamental_data['income_statement'])
            if not income_metrics.empty:
                all_metrics.append(income_metrics)
        
        if 'balance_sheet' in fundamental_data:
            balance_metrics = self.process_balance_sheet(fundamental_data['balance_sheet'])
            if not balance_metrics.empty:
                all_metrics.append(balance_metrics)
        
        if 'cash_flow' in fundamental_data:
            cashflow_metrics = self.process_cash_flow(fundamental_data['cash_flow'])
            if not cashflow_metrics.empty:
                all_metrics.append(cashflow_metrics)
        
        # Combine all metrics
        if all_metrics:
            combined_metrics = pd.concat(all_metrics, axis=1)
            
            # Add derived ratios if we have the necessary data
            if len(all_metrics) >= 2:
                income_df = all_metrics[0] if len(all_metrics) > 0 else pd.DataFrame()
                balance_df = all_metrics[1] if len(all_metrics) > 1 else pd.DataFrame()
                cashflow_df = all_metrics[2] if len(all_metrics) > 2 else pd.DataFrame()
                
                ratios = self.calculate_financial_ratios(income_df, balance_df, cashflow_df)
                if not ratios.empty:
                    combined_metrics = pd.concat([combined_metrics, ratios], axis=1)
            
            # Handle overview data separately (it's typically a single row)
            if 'overview' in fundamental_data:
                overview_metrics = self.process_company_overview(fundamental_data['overview'])
                if overview_metrics:
                    # Convert to DataFrame and broadcast to match the time series
                    overview_df = pd.DataFrame([overview_metrics] * len(combined_metrics), 
                                             index=combined_metrics.index)
                    combined_metrics = pd.concat([combined_metrics, overview_df], axis=1)
            
            return combined_metrics.fillna(method='ffill').dropna(how='all')
        
        return pd.DataFrame()
    
    def create_aggregated_fundamental_factors(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated fundamental factors using PCA
        """
        if fundamental_data.empty:
            return pd.DataFrame()
            
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        try:
            # Handle infinite values and fill NaNs
            fundamental_data_clean = fundamental_data.replace([np.inf, -np.inf], np.nan)
            fundamental_data_clean = fundamental_data_clean.fillna(method='ffill').fillna(0)
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(fundamental_data_clean)
            
            # Apply PCA to create factors
            n_components = min(5, scaled_data.shape[1])  # Limit to 5 factors for fundamentals
            pca = PCA(n_components=n_components)
            factors = pca.fit_transform(scaled_data)
            
            # Create factor DataFrame
            factor_columns = [f'Fund_Factor_{i+1}' for i in range(n_components)]
            factor_df = pd.DataFrame(factors, 
                                    index=fundamental_data.index, 
                                    columns=factor_columns)
            
            # Store the explained variance ratio
            self.fund_factor_variance = pca.explained_variance_ratio_
            self.fund_factor_components = pca.components_
            
            return factor_df
            
        except Exception as e:
            print(f"Error creating aggregated fundamental factors: {e}")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    print("Fundamental Factor Extractor module ready!")
    print("Use with financial statement data to generate fundamental factors.") 