#define TBB_PREVIEW_FLOW_GRAPH_FEATURES 1
#include "tbb/tbb_config.h"
//#include "../../common/utility/utility.h"

#if __TBB_PREVIEW_ASYNC_MSG && __TBB_CPP11_LAMBDAS_PRESENT

// includes, system
#include <iostream>
#include <stdlib.h>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>

// basic-parallel-transform.cpp
// compile with: /EHsc
#include "ipp.h"
#include "ipps.h"
#include "mkl_vsl.h"
#include <ppl.h>
#include <random>
#include <iostream>
#include <windows.h>
#include <algorithm>

#include "tbb/tbb.h"
#include "tbb/flow_graph.h"
#include "tbb/tick_count.h"
#include "tbb/concurrent_queue.h"
#include "tbb/compat/thread"
#include "bzlib.h"
#include <thread>

//#define __TBB_PREVIEW_OPENCL_NODE 1
//#include "tbb/flow_graph_opencl_node.h"

#include <numeric>

//https://software.intel.com/en-us/videos/cpus-gpus-fpgas-managing-the-alphabet-soup-with-intel-threading-building-blocks

using namespace concurrency;
using namespace std;

using namespace tbb::flow;


struct square {
	int operator()(int v) { return v*v; }
};

struct cube {
	int operator()(int v) { return v*v*v; }
};

class sum {
	int &my_sum;
public:
	sum(int &s) : my_sum(s) {}
	int operator()(tuple< int, int > v) {
		my_sum += get<0>(v) + get<1>(v);
		return my_sum;
	}
};

int tbb_join_node_example() {
	int result = 0;

	graph g;
	broadcast_node<int>     input(g);
	function_node<int, int> squarer(g, unlimited, square());
	function_node<int, int> cuber(g, unlimited, cube());
	join_node< tuple<int, int>, queueing > join(g);
	function_node< tuple<int, int>, int> summer(g, serial, sum(result));

	make_edge(input, squarer);
	make_edge(input, cuber);
	make_edge(squarer, get<0>(join.input_ports()));
	make_edge(cuber,   get<1>(join.input_ports()));
	make_edge(join, summer);

	for (int i = 1; i <= 10; ++i)
		input.try_put(i);
	g.wait_for_all();

	printf("Final result: sum of squares and cubes is %d\n", result);
	return 0;
}

typedef multifunction_node<int, tbb::flow::tuple<int, int> > multi_node;

struct MultiBody {

	void operator()(const int &i, multi_node::output_ports_type &op) {
		if (i % 2)
			std::get<1>(op).try_put(i); // put to odd queue
		else
			std::get<0>(op).try_put(i); // put to even queue
	}
};

int tbb_graph_multifunction_example() {
	graph g;

	queue_node<int> even_queue(g);
	queue_node<int> odd_queue(g);

	multi_node node1(g, unlimited, MultiBody());

	//output_port<0>(node1).register_successor(even_queue);
	make_edge(output_port<0>(node1), even_queue);		// VjD can we do this instead of the register_successor() line above, since this makes so much more sense?
	make_edge(output_port<1>(node1), odd_queue);

	for (int i = 0; i < 1000; ++i) {
		node1.try_put(i);
	}
	g.wait_for_all();

	return 0;
}

void broadcastNodeExample()
{
	tbb::flow::graph g;

	// TODO: Create a TBB working examples blog entry
	// TODO: Replace continue_msg with int or some other type to test/show that any type works to pass along within a graph
	struct body {
		std::string my_name;
		body(const char *name) : my_name(name) {}
		void operator()(continue_msg) const {
			cout << my_name << endl;
		}
	};

	broadcast_node<continue_msg> start(g);
	continue_node<continue_msg> a(g, body("A"));
	continue_node<continue_msg> b(g, body("B"));
	continue_node<continue_msg> c(g, body("C"));
	continue_node<continue_msg> d(g, body("D"));
	continue_node<continue_msg> e(g, body("E"));

	make_edge(start, a);
	make_edge(start, b);
	make_edge(a, c);
	make_edge(b, c);
	make_edge(c, d);
	make_edge(a, e);

	for (int i = 0; i < 3; ++i) {
		start.try_put(continue_msg());
		g.wait_for_all();
	}

	cout << "broadcastNodeExample: Ran successfully" << endl;
}

void broadcastNodeExample_2()
{
	tbb::flow::graph g;

	// TODO: Create a TBB working examples blog entry
	// TODO: Replace continue_msg with int or some other type to test/show that any type works to pass along within a graph
	struct body {
		std::string my_name;
		body(const char *name) : my_name(name) {}
		void operator()(continue_msg) const {
			cout << my_name << endl;
		}
	};

	broadcast_node<continue_msg> start(g);
	continue_node<continue_msg> a(g, body("A"));	// continue_node seems to take only continue_msg, which seems very limited
	continue_node<continue_msg> b(g, body("B"));
	continue_node<continue_msg> c(g, body("C"));
	continue_node<continue_msg> d(g, body("D"));
	continue_node<continue_msg> e(g, body("E"));

	make_edge(start, a);
	make_edge(start, b);
	make_edge(a, c);
	make_edge(b, c);
	make_edge(c, d);
	make_edge(a, e);

	for (int i = 0; i < 3; ++i) {
		start.try_put(continue_msg());
		g.wait_for_all();
	}

	cout << "broadcastNodeExample: Ran successfully" << endl;
}

// TODO: Split this into a separate source file!
int tbbGraphExample()
{
	tbb::flow::graph g;
	tbb::flow::continue_node< tbb::flow::continue_msg > h(g,
		[](const tbb::flow::continue_msg &) {
		cout << "Hello Flow Graph ";
	});
	tbb::flow::continue_node< tbb::flow::continue_msg > w(g,
		[](const tbb::flow::continue_msg &) {
		cout << "World\n";
	});
	tbb::flow::make_edge(h, w);
	h.try_put(tbb::flow::continue_msg());
	g.wait_for_all();
	return 1;
}

int indexerNodeExample()
{
	graph g;
	function_node<int,   int>   f1(g, unlimited, [](const int   &i) { return 2 * i; });
	function_node<float, float> f2(g, unlimited, [](const float &f) { return f / 2; });

	typedef indexer_node<int, float> my_indexer_type;
	my_indexer_type indexer(g);

	function_node< my_indexer_type::output_type > f3(g, unlimited, [](const my_indexer_type::output_type &v) {
		if (v.tag() == 0) {
			printf("Received an int %d\n", cast_to<int>(v));
		}
		else {
			printf("Received a float %f\n", cast_to<float>(v));
		}
	});

	make_edge(f1, input_port<0>(indexer));
	make_edge(f2, input_port<1>(indexer));
	make_edge(indexer, f3);

	f1.try_put(3);
	f2.try_put(3);
	g.wait_for_all();
	return 0;
}

// TODO: Implement indexer_node with output and put a queue_node on output and pull from that queue with code outside of graph
int indexerNodeExampleWithOutputAndQueue()
{
	graph g;
	function_node<int,   int>   f1(g, unlimited, [](const int   &i) { return 2 * i; });
	function_node<float, float> f2(g, unlimited, [](const float &f) { return f / 2; });

	typedef indexer_node<int, float> my_indexer_type;
	my_indexer_type indexer(g);
	queue_node<my_indexer_type::output_type> indexer_queue(g);

	function_node< my_indexer_type::output_type > f3(g, unlimited, [](const my_indexer_type::output_type &v) {
		if (v.tag() == 0) {
			printf("Received an int %d\n", cast_to<int>(v));
		}
		else {
			printf("Received a float %f\n", cast_to<float>(v));
		}
	});

	make_edge(f1, input_port<0>(indexer));
	make_edge(f2, input_port<1>(indexer));
	make_edge(indexer, indexer_queue);

	f1.try_put(3);	// send inputs
	f2.try_put(3);

	// Collect outputs
	my_indexer_type::output_type result;
	unsigned numResultsExpected = 2;
	for( unsigned i = 0; i < numResultsExpected;)
	{
		if (indexer_queue.try_get(result))
		{
			i++;
			if (result.tag() == 0) {
				printf("Received an int %d\n", cast_to<int>(result));
			}
			else {
				printf("Received a float %f\n", cast_to<float>(result));
			}
		}
	}

	g.wait_for_all();
	return 0;
}

#if 0
int opencl_asyncExample()
{
#if 1
	//// OpenCL Node TBB Graph experimentation
	//using namespace tbb::flow;
	//opencl_graph g;

	//opencl_node < tuple<opencl_buffer<cl_char>>> clPrint(g, "hello_world.cl", "print");

	//const char str[] = "Hello, World!";
	//opencl_buffer<cl_char> b(g, sizeof(str));
	//std::copy_n(str, sizeof(str), b.begin());

	//clPrint.set_ndranges({ 1 });
	////clPrint.opencl_range({ 1 });

	//input_port<0>(clPrint).try_put(b);

#else
    opencl_graph g;
    opencl_node<tuple<cl_int>> cl1( g, "simple_dependency.cl", "k1" );
    opencl_node<tuple<cl_int>> cl2( g, "simple_dependency.cl", "k2" );
    opencl_node<tuple<cl_int>> cl3( g, "simple_dependency.cl", "k3" );
         
    make_edge( output_port<0>(cl1), input_port<0>(cl2) );
    make_edge( output_port<0>(cl1), input_port<0>(cl3) );
  
    cl1.set_ndranges( { 1 } );
    cl2.set_ndranges( { 1 } );
    cl3.set_ndranges( { 1 } );
    input_port<0>(cl1).try_put( 0 );
#endif
	g.wait_for_all();

	return 0;
}
#endif

#endif