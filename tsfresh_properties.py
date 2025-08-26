ENHANCED_TSFRESH_CONFIG = {
    # Categorical risk feature (1-8 scale) - Enhanced with behavioral patterns
    "risk_category": {
        # Advanced trend analysis with multiple time windows
        "linear_trend": [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "slope"}, {"attr": "stderr"}],
        "agg_linear_trend": [
            {"attr": "pvalue", "chunk_len": 3, "f_agg": "min"},   # Short-term significance
            {"attr": "pvalue", "chunk_len": 6, "f_agg": "min"},   # Medium-term significance
            {"attr": "slope", "chunk_len": 3, "f_agg": "mean"},   # Short-term trend
            {"attr": "slope", "chunk_len": 6, "f_agg": "mean"},   # Medium-term trend
            {"attr": "slope", "chunk_len": 3, "f_agg": "var"},    # Trend volatility
        ],
        
        # Enhanced statistical measures with higher moments
        "mean": None,
        "median": None,
        "std": None,
        "var": None,
        "skewness": None,
        "kurtosis": None,
        "root_mean_square": None,  # RMS for overall magnitude
        
        # Distribution and risk zone analysis
        "minimum": None,
        "maximum": None,
        "range_count": [
            {"min": 1, "max": 2},   # Low risk
            {"min": 3, "max": 4},   # Medium-low risk
            {"min": 5, "max": 5},   # Medium risk
            {"min": 6, "max": 7},   # High risk
            {"min": 8, "max": 8}    # Critical risk
        ],
        "value_count": [{"value": i} for i in range(1, 9)],  # All risk levels
        
        # Advanced change pattern analysis
        "mean_change": None,
        "mean_abs_change": None,
        "absolute_sum_of_changes": None,
        "mean_second_derivative_central": None,  # Acceleration/deceleration
        "c3": [{"lag": 1}, {"lag": 2}, {"lag": 3}],  # Non-linearity measure
        
        # Risk persistence patterns
        "longest_strike_above_mean": None,
        "longest_strike_below_mean": None,
        "count_above_mean": None,
        "count_below_mean": None,
        
        # Enhanced quantile analysis
        "quantile": [{"q": 0.05}, {"q": 0.1}, {"q": 0.25}, {"q": 0.75}, {"q": 0.9}, {"q": 0.95}],
        
        # Risk transition detection
        "number_crossing_m": [{"m": 4}, {"m": 5}, {"m": 6}],  # Crossing medium-high risk thresholds
        "last_location_of_maximum": None,
        "first_location_of_maximum": None,
        "last_location_of_minimum": None,
        "first_location_of_minimum": None,
        
        # Volatility and stability measures
        "variance_larger_than_standard_deviation": None,
        "percentage_of_reoccurring_values_to_all_values": None,
        "percentage_of_reoccurring_datapoints_to_all_datapoints": None,
        
        # Fourier features for cyclical patterns
        "fft_coefficient": [{"coeff": 0, "attr": "real"}, {"coeff": 0, "attr": "imag"}, 
                           {"coeff": 1, "attr": "real"}, {"coeff": 1, "attr": "imag"}],
        
        # Energy and complexity measures
        "sum_of_reoccurring_values": None,
        "sum_of_reoccurring_data_points": None,
        "ratio_beyond_r_sigma": [{"r": 1}, {"r": 1.5}, {"r": 2}, {"r": 2.5}],
    },
    
    # Billing balance - Enhanced with financial behavior analysis
    "billing_balance": {
        # Core statistical features with financial context
        "mean": None,
        "median": None,
        "std": None,
        "var": None,
        "skewness": None,
        "kurtosis": None,
        "root_mean_square": None,
        
        # Advanced trend analysis for spending patterns
        "linear_trend": [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "slope"}, {"attr": "stderr"}],
        "agg_linear_trend": [
            {"attr": "slope", "chunk_len": 3, "f_agg": "mean"},   # Short-term spending trend
            {"attr": "slope", "chunk_len": 6, "f_agg": "mean"},   # Medium-term spending trend
            {"attr": "slope", "chunk_len": 12, "f_agg": "mean"},  # Long-term spending trend
            {"attr": "pvalue", "chunk_len": 3, "f_agg": "min"},   # Trend significance
            {"attr": "slope", "chunk_len": 3, "f_agg": "var"},    # Spending volatility
        ],
        
        # Change and volatility with financial context
        "mean_change": None,
        "mean_abs_change": None,
        "absolute_sum_of_changes": None,
        "mean_second_derivative_central": None,  # Balance acceleration
        "variance_larger_than_standard_deviation": None,
        
        # Distribution characteristics
        "minimum": None,
        "maximum": None,
        "quantile": [{"q": 0.05}, {"q": 0.1}, {"q": 0.25}, {"q": 0.5}, 
                    {"q": 0.75}, {"q": 0.9}, {"q": 0.95}],
        
        # Spending pattern detection
        "number_peaks": [{"n": 1}, {"n": 2}, {"n": 3}, {"n": 5}],
        "number_cwt_peaks": [{"n": 1}, {"n": 2}, {"n": 3}, {"n": 5}],
        "longest_strike_above_mean": None,
        "longest_strike_below_mean": None,
        
        # Zero/negative balance analysis
        "count_below_mean": None,
        "count_above_mean": None,
        "ratio_beyond_r_sigma": [{"r": 1}, {"r": 1.5}, {"r": 2}, {"r": 2.5}],
        "number_crossing_m": [{"m": 0}],  # Zero balance crossings
        
        # Autocorrelation for spending cycles
        "autocorrelation": [{"lag": 1}, {"lag": 2}, {"lag": 3}, {"lag": 6}, {"lag": 12}],
        "partial_autocorrelation": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
        
        # Balance utilization patterns
        "percentage_of_reoccurring_datapoints_to_all_datapoints": None,
        "sum_of_reoccurring_values": None,
        "sum_of_reoccurring_data_points": None,
        
        # Fourier analysis for seasonal spending
        "fft_coefficient": [{"coeff": 0, "attr": "real"}, {"coeff": 0, "attr": "imag"},
                           {"coeff": 1, "attr": "real"}, {"coeff": 1, "attr": "imag"}],
        
        # Financial stress indicators
        "benford_correlation": None,  # Natural spending distribution
        "approximate_entropy": [{"m": 2, "r": 0.1}],  # Spending predictability
        "sample_entropy": None,
        
        # Time-based location features
        "last_location_of_maximum": None,
        "first_location_of_maximum": None,
        "last_location_of_minimum": None,
        "first_location_of_minimum": None,
    },
    
    # Minimum due payment - Enhanced payment behavior analysis
    "minimum_due": {
        # Statistical measures
        "mean": None,
        "median": None,
        "std": None,
        "var": None,
        "minimum": None,
        "maximum": None,
        "root_mean_square": None,
        
        # Change patterns with payment context
        "mean_change": None,
        "mean_abs_change": None,
        "absolute_sum_of_changes": None,
        "mean_second_derivative_central": None,  # Payment amount acceleration
        
        # Enhanced trend analysis
        "linear_trend": [{"attr": "slope"}, {"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "stderr"}],
        "agg_linear_trend": [
            {"attr": "slope", "chunk_len": 3, "f_agg": "mean"},
            {"attr": "slope", "chunk_len": 6, "f_agg": "mean"},
            {"attr": "pvalue", "chunk_len": 3, "f_agg": "min"},
        ],
        
        # Distribution analysis
        "quantile": [{"q": 0.1}, {"q": 0.25}, {"q": 0.5}, {"q": 0.75}, {"q": 0.9}],
        "skewness": None,
        "kurtosis": None,
        
        # Payment behavior patterns
        "longest_strike_above_mean": None,
        "longest_strike_below_mean": None,
        "count_above_mean": None,
        "count_below_mean": None,
        "ratio_beyond_r_sigma": [{"r": 1}, {"r": 1.5}, {"r": 2}],
        
        # Zero/minimal payment detection
        "has_duplicate_min": None,
        "has_duplicate_max": None,
        "value_count": [{"value": 0}],  # Zero minimum due
        "ratio_value_number_to_time_series_length": [{"value": 0}],
        "percentage_of_reoccurring_values_to_all_values": None,
        
        # Payment volatility
        "variance_larger_than_standard_deviation": None,
        "number_peaks": [{"n": 1}, {"n": 2}],
        
        # Autocorrelation for payment patterns
        "autocorrelation": [{"lag": 1}, {"lag": 2}],
        
        # Crossing analysis
        "number_crossing_m": [{"m": 0}],  # Zero crossings
        
        # Energy measures
        "sum_of_reoccurring_values": None,
        "sum_of_reoccurring_data_points": None,
    },
    
    # Days past due - Critical delinquency analysis
    "days_past_due": {
        # Critical statistical measures
        "mean": None,
        "median": None,
        "std": None,
        "var": None,
        "maximum": None,
        "minimum": None,
        "root_mean_square": None,
        
        # Enhanced delinquency trend analysis
        "linear_trend": [{"attr": "slope"}, {"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "stderr"}],
        "agg_linear_trend": [
            {"attr": "slope", "chunk_len": 3, "f_agg": "mean"},   # Short-term delinquency trend
            {"attr": "slope", "chunk_len": 6, "f_agg": "mean"},   # Medium-term delinquency trend
            {"attr": "pvalue", "chunk_len": 3, "f_agg": "min"},   # Trend significance
            {"attr": "slope", "chunk_len": 3, "f_agg": "var"},    # Delinquency volatility
        ],
        
        # Change and escalation patterns
        "mean_change": None,
        "mean_abs_change": None,
        "absolute_sum_of_changes": None,
        "mean_second_derivative_central": None,  # Delinquency acceleration
        
        # Enhanced distribution and risk buckets
        "quantile": [{"q": 0.25}, {"q": 0.5}, {"q": 0.75}, {"q": 0.9}, {"q": 0.95}, {"q": 0.99}],
        "range_count": [
            {"min": 0, "max": 0},     # Current (exactly 0)
            {"min": 1, "max": 30},    # 1-30 days
            {"min": 31, "max": 60},   # 31-60 days  
            {"min": 61, "max": 90},   # 61-90 days
            {"min": 91, "max": 120},  # 91-120 days
            {"min": 121, "max": 180}, # 121-180 days
            {"min": 181, "max": 1000} # 180+ days (charge-off territory)
        ],
        
        # Delinquency behavior patterns
        "longest_strike_above_mean": None,
        "longest_strike_below_mean": None,
        "count_above_mean": None,
        "count_below_mean": None,
        "ratio_beyond_r_sigma": [{"r": 1}, {"r": 1.5}, {"r": 2}, {"r": 3}],
        
        # Current account analysis
        "ratio_value_number_to_time_series_length": [{"value": 0}],
        "value_count": [{"value": 0}],  # Count of current periods
        "number_crossing_m": [{"m": 0}, {"m": 30}, {"m": 60}, {"m": 90}],  # Bucket transitions
        
        # Peak delinquency detection
        "number_peaks": [{"n": 1}, {"n": 2}, {"n": 3}],
        "number_cwt_peaks": [{"n": 1}, {"n": 2}],
        "last_location_of_maximum": None,
        "first_location_of_maximum": None,
        "last_location_of_minimum": None,
        "first_location_of_minimum": None,
        
        # Payment cycle analysis
        "autocorrelation": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
        "partial_autocorrelation": [{"lag": 1}, {"lag": 2}],
        
        # Advanced delinquency measures
        "variance_larger_than_standard_deviation": None,
        "percentage_of_reoccurring_datapoints_to_all_datapoints": None,
        "percentage_of_reoccurring_values_to_all_values": None,
        
        # Complexity and entropy measures
        "approximate_entropy": [{"m": 2, "r": 0.1}],  # Payment behavior predictability
        "sample_entropy": None,
        "sum_of_reoccurring_values": None,
        "sum_of_reoccurring_data_points": None,
        
        # Higher-order statistics
        "skewness": None,
        "kurtosis": None,
        
        # Non-linear measures
        "c3": [{"lag": 1}, {"lag": 2}],  # Third-order cumulant
        
        # Fourier analysis for cyclical delinquency patterns
        "fft_coefficient": [{"coeff": 0, "attr": "real"}, {"coeff": 0, "attr": "imag"}],
        
        # Benford's law for natural delinquency distribution
        "benford_correlation": None,
    }
}

