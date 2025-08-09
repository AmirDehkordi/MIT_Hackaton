"""
Backtesting Framework for the Forecasting System
Evaluates model performance using walk-forward analysis and various metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from main_forecasting_system import ComprehensiveStockForecaster
import config

warnings.filterwarnings('ignore')

class ForecastingBacktester:
    def __init__(self, tickers: List[str], start_date: str = "2020-01-01", end_date: str = None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        self.backtest_results = {}
        self.performance_metrics = {}
        self.trading_results = {}
        
    def walk_forward_backtest(self, ticker: str, window_size: int = 252, 
                            step_size: int = 22, min_train_size: int = 504) -> Dict:
        """
        Perform walk-forward backtesting for a single ticker
        """
        print(f"Starting walk-forward backtest for {ticker}...")
        
        forecaster = ComprehensiveStockForecaster(ticker)
        
        # Collect all data first
        all_data = forecaster.collect_all_data(force_refresh=True)
        price_data = all_data['price_data']
        
        # Define backtesting periods
        total_periods = len(price_data)
        start_idx = min_train_size
        
        predictions = []
        actuals = []
        dates = []
        signals = []
        confidence_scores = []
        
        backtest_log = []
        
        while start_idx + window_size < total_periods:
            try:
                # Define training and testing periods
                train_end_idx = start_idx
                test_start_idx = start_idx
                test_end_idx = min(start_idx + step_size, total_periods)
                
                # Get training data
                train_data = price_data.iloc[:train_end_idx]
                test_data = price_data.iloc[test_start_idx:test_end_idx]
                
                print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
                print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")
                
                # Create forecaster with training data only
                forecaster_bt = ComprehensiveStockForecaster(ticker)
                
                # Override the data collection to use only training data
                forecaster_bt.raw_data = {
                    'price_data': train_data,
                    'fundamental_data': all_data['fundamental_data'],
                    'market_data': all_data['market_data']
                }
                
                # Run the forecasting pipeline
                factors = forecaster_bt.extract_all_factors()
                var_results = forecaster_bt.build_var_model()
                ml_results = forecaster_bt.train_ml_models()
                forecast = forecaster_bt.generate_comprehensive_forecast(forecast_days=step_size)
                
                # Get predictions and actuals
                forecast_df = forecast['forecast_df']
                
                # Align predictions with actual test data
                for i, (pred_date, actual_date) in enumerate(zip(forecast_df.index, test_data.index)):
                    if i < len(forecast_df) and i < len(test_data):
                        pred_price = forecast_df.iloc[i]['Predicted_Price']
                        actual_price = test_data.iloc[i]['Close']
                        
                        predictions.append(pred_price)
                        actuals.append(actual_price)
                        dates.append(actual_date)
                        signals.append(forecast_df.iloc[i]['Signal'])
                        confidence_scores.append(forecast_df.iloc[i]['Confidence_Score'])
                
                # Log this backtest iteration
                backtest_log.append({
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'num_predictions': len(forecast_df),
                    'model_performance': forecaster_bt.model_performance
                })
                
                print(f"Completed backtest iteration {len(backtest_log)}")
                
            except Exception as e:
                print(f"Error in backtest iteration: {e}")
                
            # Move to next period
            start_idx += step_size
        
        # Compile results
        backtest_results = {
            'ticker': ticker,
            'dates': dates,
            'predictions': predictions,
            'actuals': actuals,
            'signals': signals,
            'confidence_scores': confidence_scores,
            'backtest_log': backtest_log
        }
        
        self.backtest_results[ticker] = backtest_results
        
        print(f"Walk-forward backtest completed for {ticker}")
        return backtest_results
    
    def calculate_performance_metrics(self, ticker: str) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if ticker not in self.backtest_results:
            raise ValueError(f"No backtest results for {ticker}")
        
        results = self.backtest_results[ticker]
        predictions = np.array(results['predictions'])
        actuals = np.array(results['actuals'])
        
        # Price prediction metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals, predictions)
        
        # Percentage errors
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Directional accuracy
        actual_returns = np.diff(actuals) / actuals[:-1]
        predicted_returns = np.diff(predictions) / actuals[:-1]  # Use actual prices as base
        
        direction_correct = np.sum((actual_returns > 0) == (predicted_returns > 0))
        directional_accuracy = direction_correct / len(actual_returns)
        
        # Volatility metrics
        actual_volatility = np.std(actual_returns)
        predicted_volatility = np.std(predicted_returns)
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'Actual_Volatility': actual_volatility,
            'Predicted_Volatility': predicted_volatility,
            'Volatility_Ratio': predicted_volatility / actual_volatility if actual_volatility > 0 else np.nan
        }
        
        self.performance_metrics[ticker] = metrics
        
        return metrics
    
    def simulate_trading_strategy(self, ticker: str, initial_capital: float = 100000,
                                transaction_cost: float = 0.001) -> Dict:
        """
        Simulate a trading strategy based on the model signals
        """
        if ticker not in self.backtest_results:
            raise ValueError(f"No backtest results for {ticker}")
        
        results = self.backtest_results[ticker]
        dates = results['dates']
        actuals = results['actuals']
        signals = results['signals']
        confidence_scores = results['confidence_scores']
        
        # Trading simulation
        capital = initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        shares = 0
        portfolio_values = [initial_capital]
        trade_log = []
        
        for i in range(len(dates)):
            current_price = actuals[i]
            signal = signals[i]
            confidence = confidence_scores[i]
            
            # Simple trading logic based on signals and confidence
            if signal in ['Strong Buy', 'Buy'] and confidence > 0.3 and position <= 0:
                # Buy signal
                if position == -1:
                    # Close short position
                    capital += shares * current_price * (1 - transaction_cost)
                    trade_log.append({
                        'date': dates[i],
                        'action': 'Cover Short',
                        'price': current_price,
                        'shares': -shares,
                        'capital': capital
                    })
                
                # Open long position
                shares = int(capital * 0.95 / current_price)  # Use 95% of capital
                capital -= shares * current_price * (1 + transaction_cost)
                position = 1
                
                trade_log.append({
                    'date': dates[i],
                    'action': 'Buy',
                    'price': current_price,
                    'shares': shares,
                    'capital': capital
                })
                
            elif signal in ['Strong Sell', 'Sell'] and confidence > 0.3 and position >= 0:
                # Sell signal
                if position == 1:
                    # Close long position
                    capital += shares * current_price * (1 - transaction_cost)
                    trade_log.append({
                        'date': dates[i],
                        'action': 'Sell',
                        'price': current_price,
                        'shares': shares,
                        'capital': capital
                    })
                
                # Open short position (if allowed)
                shares = -int(capital * 0.95 / current_price)  # Negative for short
                capital -= abs(shares) * current_price * (1 + transaction_cost)
                position = -1
                
                trade_log.append({
                    'date': dates[i],
                    'action': 'Short',
                    'price': current_price,
                    'shares': shares,
                    'capital': capital
                })
            
            # Calculate portfolio value
            if position == 1:
                portfolio_value = capital + shares * current_price
            elif position == -1:
                portfolio_value = capital - shares * current_price  # shares is negative for short
            else:
                portfolio_value = capital
            
            portfolio_values.append(portfolio_value)
        
        # Final portfolio value
        final_portfolio_value = portfolio_values[-1]
        total_return = (final_portfolio_value - initial_capital) / initial_capital
        
        # Calculate buy-and-hold return for comparison
        if len(actuals) > 0:
            buy_hold_return = (actuals[-1] - actuals[0]) / actuals[0]
        else:
            buy_hold_return = 0
        
        # Calculate additional metrics
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        if len(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        trading_results = {
            'ticker': ticker,
            'initial_capital': initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trade_log),
            'portfolio_values': portfolio_values,
            'trade_log': trade_log,
            'dates': [dates[0]] + dates  # Add initial date for portfolio values
        }
        
        self.trading_results[ticker] = trading_results
        
        return trading_results
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """
        Calculate maximum drawdown
        """
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def run_comprehensive_backtest(self) -> Dict:
        """
        Run comprehensive backtest for all tickers
        """
        print("Starting comprehensive backtesting...")
        
        results_summary = {}
        
        for ticker in self.tickers:
            print(f"\n{'='*50}")
            print(f"Backtesting {ticker}")
            print(f"{'='*50}")
            
            try:
                # Run walk-forward backtest
                backtest_results = self.walk_forward_backtest(ticker)
                
                # Calculate performance metrics
                performance_metrics = self.calculate_performance_metrics(ticker)
                
                # Simulate trading strategy
                trading_results = self.simulate_trading_strategy(ticker)
                
                results_summary[ticker] = {
                    'backtest_completed': True,
                    'performance_metrics': performance_metrics,
                    'trading_results': trading_results
                }
                
                print(f"\nResults for {ticker}:")
                print(f"R²: {performance_metrics['R2']:.4f}")
                print(f"MAPE: {performance_metrics['MAPE']:.2f}%")
                print(f"Directional Accuracy: {performance_metrics['Directional_Accuracy']:.2%}")
                print(f"Total Return: {trading_results['total_return']:.2%}")
                print(f"Buy-Hold Return: {trading_results['buy_hold_return']:.2%}")
                print(f"Excess Return: {trading_results['excess_return']:.2%}")
                print(f"Sharpe Ratio: {trading_results['sharpe_ratio']:.2f}")
                print(f"Max Drawdown: {trading_results['max_drawdown']:.2%}")
                
            except Exception as e:
                print(f"Error backtesting {ticker}: {e}")
                results_summary[ticker] = {
                    'backtest_completed': False,
                    'error': str(e)
                }
        
        return results_summary
    
    def create_performance_report(self, save_path: str = None) -> None:
        """
        Create comprehensive performance report with visualizations
        """
        if not self.performance_metrics or not self.trading_results:
            print("No backtest results available. Run comprehensive backtest first.")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. Performance Metrics Comparison
        ax1 = axes[0, 0]
        metrics_df = pd.DataFrame(self.performance_metrics).T
        metrics_to_plot = ['R2', 'Directional_Accuracy', 'MAPE']
        
        x_pos = np.arange(len(self.tickers))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            if metric == 'MAPE':
                # Invert MAPE for better visualization (lower is better)
                values = 100 - metrics_df[metric].values
                label = '100 - MAPE'
            else:
                values = metrics_df[metric].values
                label = metric
            
            ax1.bar(x_pos + i * width, values, width, label=label)
        
        ax1.set_xlabel('Tickers')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Metrics')
        ax1.set_xticks(x_pos + width)
        ax1.set_xticklabels(self.tickers)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Trading Performance
        ax2 = axes[0, 1]
        trading_df = pd.DataFrame({
            ticker: {
                'Total Return': results['total_return'],
                'Buy-Hold Return': results['buy_hold_return'],
                'Excess Return': results['excess_return']
            }
            for ticker, results in self.trading_results.items()
        }).T
        
        trading_df.plot(kind='bar', ax=ax2, rot=45)
        ax2.set_title('Trading Strategy Performance')
        ax2.set_ylabel('Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Portfolio Value Evolution (first ticker as example)
        ax3 = axes[1, 0]
        first_ticker = list(self.trading_results.keys())[0]
        trading_result = self.trading_results[first_ticker]
        
        dates = pd.to_datetime(trading_result['dates'])
        portfolio_values = trading_result['portfolio_values']
        
        ax3.plot(dates, portfolio_values, linewidth=2, label='Strategy')
        
        # Calculate buy-and-hold portfolio
        initial_capital = trading_result['initial_capital']
        first_price = self.backtest_results[first_ticker]['actuals'][0]
        buy_hold_values = [initial_capital * (price / first_price) 
                          for price in [first_price] + self.backtest_results[first_ticker]['actuals']]
        
        ax3.plot(dates, buy_hold_values, linewidth=2, label='Buy & Hold', linestyle='--')
        ax3.set_title(f'Portfolio Evolution - {first_ticker}')
        ax3.set_ylabel('Portfolio Value ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Prediction vs Actual (first ticker)
        ax4 = axes[1, 1]
        backtest_result = self.backtest_results[first_ticker]
        sample_size = min(100, len(backtest_result['predictions']))  # Show last 100 predictions
        
        predictions = backtest_result['predictions'][-sample_size:]
        actuals = backtest_result['actuals'][-sample_size:]
        dates_sample = pd.to_datetime(backtest_result['dates'][-sample_size:])
        
        ax4.plot(dates_sample, actuals, label='Actual', linewidth=2)
        ax4.plot(dates_sample, predictions, label='Predicted', linewidth=2, alpha=0.7)
        ax4.set_title(f'Predictions vs Actual - {first_ticker} (Last {sample_size} points)')
        ax4.set_ylabel('Price ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Risk-Return Scatter
        ax5 = axes[2, 0]
        returns = [results['total_return'] for results in self.trading_results.values()]
        volatilities = [self.performance_metrics[ticker]['Predicted_Volatility'] 
                       for ticker in self.trading_results.keys()]
        
        scatter = ax5.scatter(volatilities, returns, 
                             c=range(len(self.tickers)), cmap='viridis', s=100)
        
        for i, ticker in enumerate(self.tickers):
            ax5.annotate(ticker, (volatilities[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax5.set_xlabel('Predicted Volatility')
        ax5.set_ylabel('Total Return')
        ax5.set_title('Risk-Return Profile')
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary Statistics Table
        ax6 = axes[2, 1]
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create summary table
        summary_data = []
        for ticker in self.tickers:
            if ticker in self.performance_metrics and ticker in self.trading_results:
                summary_data.append([
                    ticker,
                    f"{self.performance_metrics[ticker]['R2']:.3f}",
                    f"{self.performance_metrics[ticker]['Directional_Accuracy']:.2%}",
                    f"{self.trading_results[ticker]['total_return']:.2%}",
                    f"{self.trading_results[ticker]['sharpe_ratio']:.2f}",
                    f"{self.trading_results[ticker]['max_drawdown']:.2%}"
                ])
        
        table = ax6.table(cellText=summary_data,
                         colLabels=['Ticker', 'R²', 'Dir. Acc.', 'Return', 'Sharpe', 'Max DD'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax6.set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance report saved to {save_path}")
        
        plt.show()
    
    def save_backtest_results(self, directory: str = "backtest_results") -> None:
        """
        Save all backtest results to files
        """
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save detailed results
        for ticker in self.tickers:
            if ticker in self.backtest_results:
                # Save backtest results
                backtest_df = pd.DataFrame({
                    'Date': self.backtest_results[ticker]['dates'],
                    'Actual_Price': self.backtest_results[ticker]['actuals'],
                    'Predicted_Price': self.backtest_results[ticker]['predictions'],
                    'Signal': self.backtest_results[ticker]['signals'],
                    'Confidence': self.backtest_results[ticker]['confidence_scores']
                })
                backtest_df.to_csv(f"{directory}/backtest_{ticker}.csv", index=False)
                
                # Save trading results
                if ticker in self.trading_results:
                    trading_df = pd.DataFrame(self.trading_results[ticker]['trade_log'])
                    trading_df.to_csv(f"{directory}/trades_{ticker}.csv", index=False)
        
        # Save summary metrics
        if self.performance_metrics:
            performance_df = pd.DataFrame(self.performance_metrics).T
            performance_df.to_csv(f"{directory}/performance_metrics.csv")
        
        if self.trading_results:
            trading_summary = pd.DataFrame({
                ticker: {
                    'Total_Return': results['total_return'],
                    'Buy_Hold_Return': results['buy_hold_return'],
                    'Excess_Return': results['excess_return'],
                    'Sharpe_Ratio': results['sharpe_ratio'],
                    'Max_Drawdown': results['max_drawdown'],
                    'Num_Trades': results['num_trades']
                }
                for ticker, results in self.trading_results.items()
            }).T
            trading_summary.to_csv(f"{directory}/trading_summary.csv")
        
        print(f"Backtest results saved to {directory}/")

# Example usage
if __name__ == "__main__":
    # Example: Backtest multiple stocks
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    backtester = ForecastingBacktester(tickers, start_date="2020-01-01")
    
    # Run comprehensive backtest
    results = backtester.run_comprehensive_backtest()
    
    # Create performance report
    backtester.create_performance_report(save_path="backtest_performance_report.png")
    
    # Save results
    backtester.save_backtest_results()
    
    print("Backtesting completed successfully!") 