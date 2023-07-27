#!/usr/bin/env python
from random import Random
from environment.utils.random_planar_graph import graphops
from environment.utils.random_planar_graph import graphio

def default_seed():
	import os, struct
	try:
		# get very random 32-bit int from the operating system
		return struct.unpack('I', os.urandom(4))[0]
	except NotImplementedError:
		# backup seed: this can be imperfect so we don't want it always
		import time
		return int(time.time()) | os.getpid()

def make_streams(seed):
	# since triangulator is specialised and might need its own random stream
	# may as well stream the other steps too!
	streams = {}
	i=0
	for k in ['gen', 'tri', 'span', 'ext', 'double']:
		streams[k] = Random(seed+i)
		i += 1
	return streams

def get_random_planar_graph(options=None): # Create random planar graphs
	if not options:
		options = {
			"width": 320, # "Width of the field on which to place points.  neato might choose a different width for the output image."
			"height": 240, # "Height of the field on which to place points.  As above, neato might choose a different size."
			"nodes": 10, # "Number of nodes to place."
			"edges": None, # "Number of edges to use for connections.  Double edges aren't counted."
			"radius": 40, # "Nodes will not be placed within this distance of each other."
			"double": 0.0, # "Probability of an edge being doubled."
			"hair": 0.0, # "Adjustment factor to favour dead-end nodes.  Ranges from 0.00 (least hairy) to 1.00 (most hairy).  Some dead-ends may exist even with a low hair factor."
			"seed": default_seed(), # "Seed for the random number generator."
			"debug_trimode": 'conform', # ['pyhull', 'triangle', 'conform'], "Triangulation mode to generate the initial triangular graph.  Default is conform.")
			"debug_tris": None, # "If a filename is specified here, the initial triangular graph will be saved as a graph for inspection."
			"debug_span": None, # "If a filename is specified here, the spanning tree will be saved as a graph for inspection."
		}
	# post-twiddling of arguments, and cross-checking
	if options['edges'] is None:
		options['edges'] = int(options['nodes'] * 1.25)
	options['edges'] = max(options['edges'], options['nodes']-1) # necessary to avoid a disjoint graph

	num_nodes = options['nodes']
	num_edges = options['edges']
	streams = make_streams(options['seed'])

	# first generate some points in the plane, according to our constraints
	nodes = graphops.generate_nodes(num_nodes, options['width'], options['height'], options['radius'], streams['gen'])
	num_nodes = len(nodes)
	# find a delaunay triangulation, so we have a list of edges that will give planar graphs
	tri_edges = graphops.triangulate(nodes, streams['tri'], options['debug_trimode'])
	# compute a spanning tree to ensure the graph is joined
	span_edges = graphops.spanning_tree(nodes, tri_edges, streams['span'])
	# extend the tree with some more edges to achieve our target num_edges
	# pick the extra ones from tri_edges to preserve planarity
	ext_edges = graphops.extend_edges(span_edges, num_edges, tri_edges, options['hair'], streams['ext'])
	# randomly double some edges
	if options['double'] > 0:
		ext_edges = graphops.double_up_edges(ext_edges, options['double'], streams['double'])
	# # write out to file
	# graphio.write(options['filename'], nodes, ext_edges, options['seed'])
	# write out debug traces if specified
	if options['debug_tris'] is not None:
		graphio.write(options['debug_tris'], nodes, tri_edges, options['seed'])
	if options['debug_span'] is not None:
		graphio.write(options['debug_span'], nodes, span_edges, options['seed'])
	return {
		'nodes': nodes,
		'edges': tuple(map(lambda e: (nodes[e[0]], nodes[e[1]]), ext_edges)),
		'spanning_tree': tuple(map(lambda e: (nodes[e[0]], nodes[e[1]]), span_edges)),
	}
