# -*- coding: utf-8 -*-
import math
import re
from types import SimpleNamespace

from PIL import Image, ImageFont, ImageDraw  # images
from imageio import get_writer as imageio_get_writer, imread as imageio_imread  # GIFs
from matplotlib import rc as matplotlib_rc # for regulating font
from matplotlib import use as matplotlib_use
matplotlib_use('Agg',force=True) # no display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
from seaborn import heatmap as seaborn_heatmap  # Heatmap

import pandas as pd
import numpy as np
import json
import numbers

font_dict = {'size':22}
# matplotlib_rc('font', **font_dict)

# flags = SimpleNamespace(**{
# 	"gif_speed": 0.25, # "GIF frame speed in seconds."
# 	"max_plot_size": 20, # "Maximum number of points in the plot. The smaller it is, the less RAM is required. If the log file has more than max_plot_size points, then max_plot_size means of slices are used instead."
# })
linestyle_set = ['-', '--', '-.', ':', '']
color_set = list(mcolors.TABLEAU_COLORS)

def wrap_string(s, max_len=10):
	return '\n'.join([
		s[i*max_len:(i+1)*max_len]
		for i in range(int(np.ceil(len(s)/max_len)))
	]).strip()

def line_plot(logs, figure_file, max_plot_size=20, max_length=None, show_deviation=False, base_list=None, base_shared_name='baseline', average_non_baselines=None, buckets_average='median'):
	assert not base_list or len(base_list)==len(logs), f"base_list (len {len(base_list)}) and logs (len {len(logs)}) must have same lenght or base_list should be empty"
	log_count = len(logs)
	# Get plot types
	stats = [None]*log_count
	key_ids = {}
	for i in range(log_count):
		log = logs[i]
		# Get statistics keys
		if log["length"] < 2:
			continue
		(_, obj) = log["line_example"]
		log_keys = list(obj.keys()) # statistics keys sorted by name
		for key in log_keys:
			if key not in key_ids:
				key_ids[key] = len(key_ids)
		stats[i] = log_keys
	max_stats_count = len(key_ids)
	if max_stats_count <= 0:
		print("Not enough data for a reasonable plot")
		return
	# Create new figure and two subplots, sharing both axes
	ncols=3 if max_stats_count >= 3 else max_stats_count
	nrows=math.ceil(max_stats_count/ncols)
	# First set up the figure and the axis
	# fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10)) # this method causes memory leaks
	figure = Figure(figsize=(10*ncols,7*nrows))
	canvas = FigureCanvas(figure)
	grid = GridSpec(ncols=ncols, nrows=nrows)
	axes = [figure.add_subplot(grid[id//ncols, id%ncols]) for id in range(max_stats_count)]
	# Populate axes
	lines_dict = {}
	for log_id in range(log_count):
		log = logs[log_id]
		name = log["name"]
		data_iter = log["data_iter"]
		length = log["length"]
		if length < 2:
			print(name, " has not enough data for a reasonable plot")
			continue
		print('Extracting data from:',name)
		if not max_length:
			max_length = length
		data_per_plotpoint = int(np.ceil(max_length/max_plot_size))
		plot_size = int(np.ceil(length/data_per_plotpoint))
		# Build x, y
		stat = stats[log_id]
		x = {
			key:[]
			for key in stat
		}
		y = {
			key:{
				"min":float("+inf"), 
				"max":float("-inf"), 
				"quantiles":[],
				"deviations": [],
			}
			for key in stat
		}
		last_step = 0
		for _ in range(plot_size):
			# initialize
			values = {
				key: []
				for key in stat
			}
			# compute values foreach key
			plotpoint_i = 0
			for (step, obj) in data_iter:
				if step <= last_step:
					continue
				plotpoint_i += 1
				last_step = step
				for key in stat: # foreach statistic
					v = obj.get(key,None)
					if v is not None:
						values[key].append(v)
				if plotpoint_i > data_per_plotpoint: # save plotpoint
					break
			# add average to data for plotting
			for key in stat: # foreach statistic
				value_list = values[key]
				if len(value_list) <= 0:
					continue
				stats_dict = y[key]
				stats_dict["quantiles"].append({
					'lower_quartile': float(np.quantile(value_list,0.25)), # lower quartile
					'median': float(np.quantile(value_list,0.5)), # median
					'upper_quartile': float(np.quantile(value_list,0.75)), # upper quartile
				})
				v_mean = float(np.mean(value_list))
				v_std = float(np.std(value_list))
				stats_dict["deviations"].append({
					'mean-std': v_mean-v_std,
					'mean': v_mean,
					'mean+std': v_mean+v_std,
				})
				if buckets_average == 'median':
					stats_dict["stats_to_plot"] = stats_dict["quantiles"]
				else:
					stats_dict["stats_to_plot"] = stats_dict["deviations"]
				# print(key, min(value_list))
				stats_dict["min"] = float(min(stats_dict["min"], min(value_list)))
				stats_dict["max"] = float(max(stats_dict["max"], max(value_list)))
				x[key].append(last_step)
		lines_dict[name] = {
			'x': x,
			'y': y,
			'log_id': log_id
		}
		for yk,yv in y.items():
			print('#'*10)
			# print(name)
			print(f'{yk}:', json.dumps(yv, indent=4))
			print('#'*10)
	plotted_baseline = False
	plot_dict = {}
	for name, line in lines_dict.items():
		is_baseline = base_list and name in base_list
		if plotted_baseline and is_baseline:
			continue # already plotted
		if is_baseline:
			name = base_shared_name
		# Populate axes
		x = line['x']
		y = line['y']
		log_id = line['log_id']
		stat = stats[log_id]
		plot_list = []
		for j in range(ncols):
			for i in range(nrows):
				idx = j if nrows == 1 else i*ncols+j
				if idx >= len(stat):
					continue
				key = stat[idx]
				y_key = y[key]
				x_key = x[key]
				unpack_quantiles = lambda a: map(lambda b: b.values(), a)
				y_key_lower_quartile, y_key_median, y_key_upper_quartile = map(np.array, zip(*unpack_quantiles(y_key["stats_to_plot"])))
				if base_list:
					base_line = base_list[log_id]
					base_y_key = lines_dict[base_line]['y'][key]
					base_y_key_lower_quartile, base_y_key_median, base_y_key_upper_quartile = map(np.array, zip(*unpack_quantiles(base_y_key["stats_to_plot"])))
					normalise = lambda x,y: 100*(x-y)/(y-base_y_key['min']+1)
					y_key_median = normalise(y_key_median, base_y_key_median)
					y_key_lower_quartile = normalise(y_key_lower_quartile, base_y_key_lower_quartile)
					y_key_upper_quartile = normalise(y_key_upper_quartile, base_y_key_upper_quartile)
				# print stats
				# print(f"    {key} is in [{y_key['min']},{y_key['max']}]")
				# print(f"    {key} has medians: {y_key_median}")
				# print(f"    {key} has lower quartiles: {y_key_lower_quartile}")
				# print(f"    {key} has upper quartiles: {y_key_upper_quartile}")
				if is_baseline:
					plotted_baseline = True
				plot_list.append({
					'coord': (i,j), 
					'key': key,
					'x': x_key,
					'y_q1': y_key_lower_quartile,
					'y_q2': y_key_median, 
					'y_q3': y_key_upper_quartile
				})
		plot_dict[name] = plot_list

	##############################
	##### Merge non-baselines ####
	if average_non_baselines:
		avg_fn = np.mean if average_non_baselines=='mean' else np.median
		new_plot_dict = {}
		merged_plots = {
			'coord': [],
			'key': [],
			'x': [],
			'y_q1': [],
			'y_q2': [], 
			'y_q3': []
		}
		for name, plot_list in plot_dict.items():
			is_baseline = base_list and name == base_shared_name
			if is_baseline:
				new_plot_dict[name] = plot_list
				continue
			merged_plots['coord'].append([plot['coord'] for plot in plot_list])
			merged_plots['key'].append([plot['key'] for plot in plot_list])
			merged_plots['x'].append([plot['x'] for plot in plot_list])
			merged_plots['y_q1'].append([plot['y_q1'] for plot in plot_list])
			merged_plots['y_q2'].append([plot['y_q2'] for plot in plot_list])
			merged_plots['y_q3'].append([plot['y_q3'] for plot in plot_list])
		new_plot_dict['DEER'] = [
			{
				'coord': coord,
				'key': key,
				'x': x,
				'y_q1': y_q1,
				'y_q2': y_q2,
				'y_q3': y_q3
			}
			for coord, key, x, y_q1, y_q2, y_q3 in zip(
				merged_plots['coord'][0],
				merged_plots['key'][0],
				merged_plots['x'][0],
				avg_fn(merged_plots['y_q1'], axis=0),
				avg_fn(merged_plots['y_q2'], axis=0),
				avg_fn(merged_plots['y_q3'], axis=0),
			)
		]
		plot_dict = new_plot_dict
	###############################	

	for log_id, (name, plot_list) in enumerate(plot_dict.items()):
		for plot in plot_list:
			i,j = plot['coord']
			x_key = plot['x']
			key = plot['key']
			y_key_lower_quartile = plot['y_q1']
			y_key_median = plot['y_q2']
			y_key_upper_quartile = plot['y_q3']
			# ax
			ax_id = key_ids[key]
			ax = axes[ax_id]
			format_label = lambda x: x.replace('_',' ')
			ax.set_ylabel(wrap_string(format_label(key) if not base_list else f'{format_label(key)} - % of gain over baseline', 25), fontdict=font_dict)
			ax.set_xlabel('step', fontdict=font_dict)
			# plot mean line
			ax.plot(x_key, y_key_median, label=format_label(name), linestyle=linestyle_set[log_id//len(color_set)], color=color_set[log_id%len(color_set)])
			# plot std range
			if show_deviation:
				ax.fill_between(x_key, y_key_lower_quartile, y_key_upper_quartile, alpha=0.25, color=color_set[log_id%len(color_set)])
			# show legend
			ax.legend()
			# display grid
			ax.grid(True)

	figure.savefig(figure_file,bbox_inches='tight')
	print("Plot figure saved in ", figure_file)
	figure = None

def line_plot_files(url_list, name_list, figure_file, max_step=None, max_plot_size=20, show_deviation=False, base_list=None, base_shared_name='baseline', average_non_baselines=None, statistics_list=None, buckets_average='median', step_type='num_env_steps_sampled'):
	assert len(url_list)==len(name_list), f"url_list (len {len(url_list)}) and name_list (len {len(name_list)}) must have same lenght"
	logs = []
	for url,name in zip(url_list,name_list):
		df = pd.read_csv(url)
		line_example = df.head()
		length = len(df)
		df = None
		print(f"{name} has length {length}")

		logs.append({
			'name': name, 
			'data_iter': parse(url, max_step=max_step, statistics_list=statistics_list, step_type=step_type),
			'length':length, 
			'line_example': parse_line(line_example, statistics_list=statistics_list, step_type=step_type)
		})
	max_length = max(logs, key=lambda x: x['length'])['length']
	line_plot(logs, figure_file, max_plot_size=max_plot_size, max_length=max_length, show_deviation=show_deviation, base_list=base_list, base_shared_name=base_shared_name, average_non_baselines=average_non_baselines, buckets_average=buckets_average)

def parse_line(line, statistics_list=None, step_type='num_env_steps_sampled'):
	key_list = line.columns.tolist()
	get_keys = lambda k: list(map(lambda x: x[len(k):].strip('/'), filter(lambda x: x.startswith(k), key_list)))
	get_element = lambda df, key: df[key].tolist()[0] if key in df else None
	arrayfy = lambda x: np.array(x[1:-1].split(', ') if ', ' in x[1:-1] else x[1:-1].split(' '), dtype=np.float32) if x[1:-1] else []

	step = get_element(line,f"info/{step_type}") # "num_env_steps_sampled", "num_env_steps_trained", "num_agent_steps_sampled", "num_agent_steps_trained"
	# obj = {
	# 	"median cum. reward": np.median(line["hist_stats"]["episode_reward"]),
	# 	"mean visited roads": line['custom_metrics'].get('visited_junctions_mean',line['custom_metrics'].get('visited_cells_mean',0))
	# }
	# print(get_element(line,"hist_stats/episode_reward"))
	
	obj = {
		"episode_reward_median": np.median(arrayfy(get_element(line,"hist_stats/episode_reward").replace('$',' '))),
	}
	for k in ["episode_reward_mean","episode_reward_max","episode_reward_min","episode_len_mean","episodes_total"]:
		obj[k] = get_element(line,k)
	
	agent_names = get_keys('policy_reward_mean/')
	if agent_names:
		for agent_id in agent_names:
			for k in ["policy_reward_mean","policy_reward_max","policy_reward_min"]:
				obj[f'{agent_id}_{k}'] = get_element(line,f"{k}/{agent_id}")
			obj[f"{agent_id}_policy_reward_median"] = np.median(np.median(arrayfy(get_element(line,f"hist_stats/policy_{agent_id}_reward").replace('$',' ')), axis=-1))
	else:
		agent_names = ["default_policy"]

	get_label = (lambda i,x: f"{i}_{x}") if len(agent_names) > 1 else (lambda i,x: x)
	for agent_id in agent_names:
		obj.update({
			f"info/learner/{agent_id}/{k}": get_element(line,f"info/learner/{agent_id}/{k}")
			for k in get_keys(f"info/learner/{agent_id}/")
			if isinstance(get_element(line,f"info/learner/{agent_id}/{k}"), numbers.Number)
		})
		obj.update({
			f"buffer/{agent_id}/{k}": get_element(line,f"buffer/{agent_id}/{k}")
			for k in get_keys(f"buffer/{agent_id}/")
			if isinstance(get_element(line,f"buffer/{agent_id}/{k}"), numbers.Number)
		})
	obj.update({
		f"custom_metrics/{k}": get_element(line,f"custom_metrics/{k}")
		for k in get_keys("custom_metrics/")
		if isinstance(get_element(line,f"custom_metrics/{k}"), numbers.Number)
	})

	if statistics_list:
		statistics_list = set(statistics_list)
		obj = dict(filter(lambda x: x[0] in statistics_list, obj.items()))
	return (step, obj)
	
def parse(url, max_step=None, statistics_list=None, step_type='num_env_steps_sampled'):
	with pd.read_csv(url, chunksize=1) as chunk_iter:
		for i,df in enumerate(chunk_iter):
			step, obj = parse_line(df.loc[[i]], statistics_list=statistics_list, step_type=step_type)
			if max_step and int(step) > max_step:
				return
			yield (step, obj)
		
	
def heatmap(heatmap, figure_file):
	# fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10)) # this method causes memory leaks
	figure = Figure()
	canvas = FigureCanvas(figure)
	ax = figure.add_subplot(111) # nrows=1, ncols=1, index=1
	seaborn_heatmap(data=heatmap, ax=ax)
	figure.savefig(figure_file,bbox_inches='tight')
	
def ascii_image(string, file_name):
	# find image size
	font = ImageFont.load_default()
	splitlines = string.splitlines()
	text_width = 0
	text_height = 0
	for line in splitlines:
		text_size = font.getsize(line) # for efficiency's sake, split only on the first newline, discard the rest
		text_width = max(text_width,text_size[0])
		text_height += text_size[1]+5
	text_width += 10
	# create image
	source_img = Image.new('RGB', (text_width,text_height), "black")
	draw = ImageDraw.Draw(source_img)
	draw.text((5, 5), string, font=font)
	source_img.save(file_name, "JPEG")
	
def combine_images(images_list, file_name):
	imgs = [ Image.open(i) for i in images_list ]
	# pick the smallest image, and resize the others to match it (can be arbitrary image shape here)
	min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
	imgs_comb = np.hstack( [np.asarray( i.resize(min_shape) ) for i in imgs] )
	# save the picture
	imgs_comb = Image.fromarray( imgs_comb )
	imgs_comb.save( file_name )
	
def rgb_array_image(array, file_name):
	img = Image.fromarray(array, 'RGB')
	img.save(file_name)
	
def make_gif(gif_path, file_list, gif_speed=0.25):
	with imageio_get_writer(gif_path, mode='I', duration=gif_speed) as writer:
		for filename in file_list:
			image = imageio_imread(filename)
			writer.append_data(image)
