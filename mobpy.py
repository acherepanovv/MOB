import pandas as pd
import numpy as np
import logging

from scipy.stats import chi2
from multiprocessing import Pool
from functools import partial
import pandas.core.algorithms as algos

logger = logging.getLogger("Mobpy")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s"))
logger.addHandler(handler)


def monotone_optimal_binning(X, Y, min_bin_size=0.05, min_bin_rate=0.001, min_p_val=0.95, start_bin=20, min_bins=2, max_bins=5):
	df = pd.DataFrame({'X': X, 'Y': Y})
	#Initial sttatistic
	df_stat = stats(df, start_bin=start_bin)
	
	#Make monotonic
	df_monotonic = make_monotonic(df_stat)
	n_bins = len(df_monotonic)
	if isinstance(df_monotonic['BUCKET'].iloc[-1], float):
		n_bins -= 1
	#Start algorith
	while n_bins > min_bins:
		min_p, bucket_min_p = compute_min_p_values(df_monotonic, min_size=min_bin_size, min_rate=min_bin_rate)
		n_bins = len(df_monotonic)
		if isinstance(df_monotonic['BUCKET'].iloc[-1], float):
			n_bins -= 1
		if min_p < min_p_val:
			indx_bot, indx_top = bucket_min_p
			df_monotonic = merge_rows(df_monotonic, indx_bot, indx_top)
		else:
			break

	n_bins = len(df_monotonic)
	if n_bins > max_bins:
		if isinstance(df_monotonic['BUCKET'].iloc[-1], float):
			max_bins = max_bins
		else:
			max_bins += 1

		while n_bins > max_bins:
			min_p, bucket_min_p = compute_min_p_values(df_monotonic, min_size=min_bin_size, min_rate=min_bin_rate)
			n_bins = len(df_monotonic)
			if isinstance(df_monotonic['BUCKET'].iloc[-1], float):
				n_bins -= 1

			indx_bot, indx_top = bucket_min_p
			df_monotonic = merge_rows(df_monotonic, indx_bot, indx_top)

	df_monotonic["IV"] = df_monotonic["IV"].sum()
	open_interval = df_monotonic["BUCKET"].values

	try:
		open_interval[0] = pd.Interval(left=-1e9, right=open_interval[0].right)
	except:
		df_monotonic['BUCKET'].iloc[0] = pd.Interval(left=-0.5, right=0.4)
		df_monotonic['BUCKET'].iloc[1] = pd.Interval(left=0.5, right=1.5)

	try:
		open_interval[-1] = pd.Interval(left=open_interval[-1].left, right=1e9)
	except:
		open_interval[-2] = pd.Interval(left=open_interval[-2].left, right=1e9)

	df_monotonic["BUCKET"] = open_interval
	return df_monotonic
	
	
def stats(df, start_bin=20):
	df_notmiss = df[["X", "Y"]][df["X"].notnull()]
	df_justmiss = df[["X", "Y"]][df["X"].isnull()]
	#Initial binning
	if df_notmiss["X"].nunique() == 2:
		df_bucket = pd.DataFrame({"X": df_notmiss["X"], "Y": df_notmiss["Y"], "BUCKET": df_notmiss["X"]})
	
	elif (len(df_notmiss[df_notmiss["X"] == 0]) + len(df_notmiss[df_notmiss["X"] == 1])) / len(df_notmiss["X"]) > 0.15:
		bins = algos.quantile(df_notmiss.X, np.linspace(0, 1, start_bin))
		bins = np.insert(bins, 0, 0.5)
		bins[1] = bins[1] - (bins[1]/2)
		df_bucket = pd.DataFrame({"X": df_notmiss["X"], "Y": df_notmiss["Y"], "BUCKET": pd.cut(df_notmiss["X"], np.unique(bins), include_lowest=True)})
	else:
		df_bucket = pd.DataFrame({"X": df_notmiss["X"], "Y": df_notmiss["Y"], "BUCKET": pd.qcut(df_notmiss["X"], start_bin)})
	
	df_stat = pd.DataFrame({}, index=[])
	
	group_data = df_bucket.groupby("BUCKET", as_index=True)
	df_stat["BUCKET"] = group_data.groups.keys()
	df_stat['MIN_VALUE'] = group_data.min()["X"].values
	df_stat['MAX_VALUE'] = group_data.max()["X"].values
	df_stat['COUNT'] = group_data.count()["X"].values
	df_stat['EVENT'] = group_data.sum()["Y"].values

	if df_justmiss.shape[0] > 0:
		df_stat_miss = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
		df_stat_miss['BUCKET'] = np.nan
		df_stat_miss['MAX_VALUE'] = np.nan
		df_stat_miss['COUNT'] = df_justmiss.count()["Y"]
		df_stat_miss['EVENT'] = df_justmiss.sum()["Y"]
		df_stat = df_stat.append(df_stat_miss, ignore_index=True)

	df_stat['SHARE'] = df_stat['COUNT']/df_stat['COUNT'].sum()
	df_stat['NON_EVENT'] = df_stat['COUNT'] - df_stat['EVENT']

	df_stat['EVENT_RATE'] = df_stat['EVENT'] / df_stat['COUNT']
	df_stat['NON_EVENT_RATE'] = df_stat['NON_EVENT'] / df_stat['COUNT']

	df_stat['DIST_EVENT'] = df_stat['EVENT'] / df_stat['EVENT'].sum()
	df_stat['DIST_NON_EVENT'] = df_stat['NON_EVENT'] / df_stat['NON_EVENT'].sum()

	df_stat['WOE'] = np.log(df_stat['DIST_NON_EVENT'] / df_stat['DIST_EVENT'])

	df_stat['WOE'] = df_stat['WOE'].fillna(0)
	df_stat['WOE'].replace(np.inf, 0, inplace=True)

	df_stat['IV'] = (df_stat['DIST_NON_EVENT'] - df_stat['DIST_EVENT']) * df_stat['WOE']

	df_stat['VAR_NAME'] = 'VAR'

	df_stat = df_stat[['VAR_NAME', 'BUCKET', 'MIN_VALUE', 'MAX_VALUE', 'SHARE', 'COUNT'
					   , 'EVENT', 'NON_EVENT', 'EVENT_RATE', 'NON_EVENT_RATE', 'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
	return df_stat


def make_monotonic(df_stat):
	df_up_down = {"increase": None, "decrease": None}
	for direction in ["increase", "decrease"]:
		df_monotonic = df_stat.copy()
		top_indx = df_monotonic.index[-1]
		if isinstance(df_monotonic["BUCKET"].iloc[-1], float):
			top_indx -= 1
		
		bot_indx = top_indx - 1
		while bot_indx >= 0:
			top_row = df_monotonic.iloc[top_indx]
			bot_row = df_monotonic.iloc[bot_indx]
			
			if direction == "increase":
				comparison = top_row["WOE"] < bot_row["WOE"]
			else:
				comparison = top_row["WOE"] > bot_row["WOE"]
				
			if comparison:
				df_monotonic = merge_rows(df_monotonic, bot_indx, top_indx)

				top_indx = df_monotonic.index[-1]
				if isinstance(df_monotonic["BUCKET"].iloc[-1], float):
					top_indx -= 1
				bot_indx = top_indx - 1
			else:
				top_indx -= 1
				bot_indx -= 1
		df_up_down[direction] = df_monotonic
	n_bins_up = len(df_up_down["increase"])
	n_bins_down = len(df_up_down["decrease"])

	if n_bins_up >= n_bins_down:
		best_direction = "increase"
	else:
		best_direction = "decrease"
		
	best_df = df_up_down[best_direction]			
	return best_df


def merge_rows(df_stat, bottom_id, top_id):
	if bottom_id not in df_stat.index:
		return df_stat
		
	if top_id not in df_stat.index:
		return df_stat
	
	df_merged = df_stat.copy()
	
	bot_row = df_merged.iloc[bottom_id]
	top_row = df_merged.iloc[top_id]
	
	try:
		left_bnd = bot_row['BUCKET'].left
		right_bnd = top_row['BUCKET'].right
	except:
		left_bnd = bot_row['BUCKET']
		right_bnd = top_row['BUCKET']
		
	merged_row = bot_row.copy()
	
	merged_row['BUCKET'] = pd.Interval(left=left_bnd, right=right_bnd)
	merged_row['MIN_VALUE'] = bot_row['MIN_VALUE']
	merged_row['MAX_VALUE'] = bot_row['MAX_VALUE']
	merged_row['COUNT'] = bot_row['COUNT'] + top_row['COUNT']
	merged_row['SHARE'] = merged_row['COUNT'] / df_merged['COUNT'].sum()
	merged_row["EVENT"] = bot_row["EVENT"] + top_row["EVENT"]
	merged_row["NON_EVENT"] = bot_row["NON_EVENT"] + top_row["NON_EVENT"]
	merged_row["EVENT_RATE"] = merged_row["EVENT"] / merged_row['COUNT']
	merged_row["NON_EVENT_RATE"] = merged_row["NON_EVENT"] / merged_row['COUNT']
	merged_row["DIST_EVENT"] = merged_row["EVENT"] / df_merged['EVENT'].sum()
	merged_row["DIST_NON_EVENT"] = merged_row["NON_EVENT"] / df_merged["NON_EVENT"].sum()
	merged_row["WOE"] = np.log(merged_row["DIST_NON_EVENT"] / merged_row["DIST_EVENT"])
	
	df_merged.iloc[bottom_id] = merged_row
	df_merged.drop(top_id, axis=0, inplace=True)
	df_merged.reset_index(inplace=True, drop=True)
	
	df_merged["IV"] = (df_merged["DIST_NON_EVENT"] - df_merged["DIST_EVENT"]) * df_merged["WOE"]

	return df_merged


def compute_min_p_values(monotonic_df, min_size, min_rate):
	df = monotonic_df.copy()
	n = len(df)
	if isinstance(df['BUCKET'].iloc[-1], float):
		n -= 1
	p_values = dict()
	for i in range(n-1):
		top_row = df.iloc[i+1]
		bot_row = df.iloc[i]
		if (bot_row['COUNT'] < 100) or (top_row['COUNT'] < 100):
			p_val = -2
			p_values[(i, i+1)] = p_val
			continue
		A = np.array([[bot_row['EVENT'], bot_row['NON_EVENT']], [top_row['EVENT'], top_row['NON_EVENT']]])
		R = np.array([bot_row['COUNT'], top_row['COUNT']])
		C = np.array([np.sum(A[:, j], axis=0) for j in range(A.shape[1])])
		N = np.sum(A)
		E = np.array([[R[i]*C[j]/N for j in range(A.shape[1])] for i in range(A.shape[0])])

		chi_2_stat = np.sum(np.power((A-E), 2) / E)
		p_val = chi2.cdf(chi_2_stat, df=1)
		
		if (bot_row['EVENT_RATE'] < min_rate) or (top_row['EVENT_RATE'] < min_rate):
			p_val -= 1
		if (bot_row['SHARE'] < min_size) or (top_row['SHARE'] < min_size):
			p_val -= 1
		p_values[(i, i+1)] = p_val
	bucket_min_p = min(p_values, key=p_values.get)
	min_p = min(p_values.values())
	return min_p, bucket_min_p


def create_woe_iv(raw_data, target, min_bin_size=0.05, min_bin_rate=0.001, min_p_val=0.95, start_bin=20, min_bins=2, max_bins=5):

	final_woe_iv = pd.DataFrame()
	woe_iv = pd.DataFrame()
	error_columns = []
	for column in raw_data.columns:
		if np.issubdtype(raw_data[column], np.number):
			try:
				woe_iv = monotone_optimal_binning(raw_data[column], target, min_bin_size, min_bin_rate, min_p_val
												  , start_bin
												  , min_bins
												  , max_bins)
				woe_iv['VAR_NAME'] = column
				final_woe_iv = final_woe_iv.append(woe_iv, ignore_index=True)
			except Exception as e1:
				logger.debug("e1: ") # + str(traceback.format_exc()))
				try:
					woe_iv = monotone_optimal_binning(raw_data[column], target, min_bin_size, min_bin_rate, min_p_val, 5, min_bins, max_bins)
					woe_iv['VAR_NAME'] = column
					final_woe_iv = final_woe_iv.append(woe_iv, ignore_index=True)
				except Exception as e2:
					logger.debug(f"e2: column {column}")# + str(traceback.format_exc()))
					error_columns.append(column)
	return final_woe_iv, error_columns


def create_woe_iv_optimal(raw_data, target, n=1):
	if n == 1:
		df, error = create_woe_iv(raw_data, target)
	if n >= 2:
		q = len(list(raw_data.columns))
		col_split = np.array_split(list(raw_data.columns), q)
		for i in range(q):
			col_split[i] = raw_data[col_split[i]]
		pool = Pool(n)
		mul_arg_func = partial(create_woe_iv, target=target)
		results = pool.map(mul_arg_func, col_split)
		pool.close()
		pool.join()

		outputsdf = [result[0] for result in results]
		outputslist = [result[1] for result in results]

		df = pd.concat(outputsdf)

		error_c = []
		for i in range(len(outputslist)):
			error_c.append(outputslist[i])

		error = [inner for outer in error_c for inner in outer]

	return df, error


def replace_feature_by_woe(data, woe_iv):
	data_woe = pd.DataFrame()
	for columns in data.columns:
		x_woe = data[columns].copy()
		var_woe_iv = woe_iv[woe_iv["VAR_NAME"] == columns]
		for i in range(len(data[columns])):
			for j in range(len(var_woe_iv["BUCKET"])):
				if np.isnan(data[columns].iloc[i]):
					x_woe[columns].iloc[i] = var_woe_iv["woe"].iloc[len(var_woe_iv['BUCKET'])-1]
					break
				elif data[columns.iloc[i]] in var_woe_iv['BUCKET'].iloc[j]:
					x_woe[columns].iloc[i] = var_woe_iv['WOE'].iloc[j]
					break
		data_woe['woe_' + columns] = x_woe[columns]
	return data_woe


def binary_search(arr, low, high, x):
	if high >= low:
		mid = low + (high - low)//2
		if arr[mid] == x:
			return mid
		elif arr[mid] > x:
			return binary_search(arr, low, mid - 1, x)
		else:
			return binary_search(arr, mid + 1, high, x)
	else:
		return low


def replace_feature_by_woe_optimal(data, df_woe_iv):
	data_woe = pd.DataFrame(index=data.index)
	for column in data.columns:
		X = data[column].values
		X_woe = np.empty_like(X, dtype=float)
		woe_iv = df_woe_iv[df_woe_iv["VAR_NAME"] == column].reset_index(drop=True)
		mask_miss = pd.isnull(X)
		if isinstance(woe_iv["BUCKET"].iloc[-1], float):
			bucket = woe_iv["BUCKET"].iloc[:-1]
			woe = woe_iv["WOE"].iloc[:-1]
			woe_nan = woe_iv["WOE"].iloc[-1]
			X_woe[mask_miss] = woe_nan
		else:
			bucket = woe_iv["BUCKET"].iloc[:]
			woe = woe_iv["WOE"].iloc[:]
		
		X_true = X[np.invert(mask_miss)]
		X_true_woe = np.empty_like(X_true, dtype=float)
		
		sorted_idx = np.argsort(X_true)
		X_sorted = X_true[sorted_idx]
		
		pred_indx = 0
		for bucket, bucket_woe in zip(bucket, woe):
			right = bucket.right + 1e-6
			last_indx = binary_search(X_sorted, low=0, high=len(X_sorted)-1, x=right)
			X_indx = sorted_idx[pred_indx:last_indx]
			X_true_woe[X_indx] = bucket_woe
			pred_indx = last_indx
		
		X_woe[np.invert(mask_miss)] = X_true_woe
		data_woe["WOE_" + column] = X_woe
	return data_woe

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	