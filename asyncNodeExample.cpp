#define TBB_PREVIEW_FLOW_GRAPH_FEATURES 1
#include "tbb/tbb_config.h"

#if __TBB_PREVIEW_ASYNC_MSG && __TBB_CPP11_LAMBDAS_PRESENT

// includes, system
#include <iostream>
#include <stdlib.h>

#include "tbb/tbb.h"
#include "tbb/flow_graph.h"
#include "tbb/tick_count.h"
#include "tbb/concurrent_queue.h"
#include "tbb/compat/thread"
#include <thread>

using namespace std;
using namespace tbb::flow;

typedef int input_type;
typedef int output_type;
typedef tbb::flow::async_node<input_type, output_type> async_node_type;

class AsyncActivity {
public:
	typedef async_node_type::gateway_type gateway_type;

	struct work_type {
		input_type input;
		gateway_type* gateway;
	};

	AsyncActivity() : m_endWork(false), service_thread(&AsyncActivity::workLoop, this) {}

	~AsyncActivity() {
		endWork();
		if (service_thread.joinable())
			service_thread.join();
	}

	// Submits a new work item to the external worker to be performed asynchronously with respect to the TBB graph
	void submitWorkItem(input_type i, gateway_type* gateway) {
		work_type w = { i, gateway };
		gateway->reserve_wait();
		my_work_queue.push(w);
	}

	bool endWork() {
		m_endWork = true;
		return m_endWork;
	}

private:
	void workLoop() {
		while (!m_endWork) {
			work_type w;
			while (my_work_queue.try_pop(w)) {
				output_type result = doWork(w.input);
				//send the result back to the graph
				w.gateway->try_put(result);
				// signal that work is done
				w.gateway->release_wait();
			}
		}
	}

	output_type doWork(input_type& v) {
		// performs the work on input converting it to output
		output_type returnValue = v + 5;	// add 5 to input value as work operation
		return returnValue;
	}

	bool m_endWork;		// must be first in the order to be initialized first in the list (this is stupid and VS should warn about this - the order of variable is not the same as the order in the init list - and this order is the one that's followed)
	tbb::concurrent_queue<work_type> my_work_queue;
	std::thread service_thread;
};

int tbbAsyncNodeExample()
{
	tbb::flow::graph g;
	AsyncActivity async_activity;

	async_node_type asyncWorker(g, unlimited,
		// user functor to initiate async processing by the worker
		[&](input_type input, async_node_type::gateway_type& gateway) {
		async_activity.submitWorkItem(input, &gateway);
	});

	const int limit = 10;
	int count = 0;

	tbb::flow::source_node<input_type> source(g, [&](input_type& v)->bool {
		/* produce data for async work */
		if (count < limit) {
			++count;
			v = count;
			return true;
		}
		else {
			return false;
		}
	});

	// Consumer which receives output type as input and consumes it by printing out details about it
	tbb::flow::async_node<output_type, output_type> workConsumer(g, unlimited, [](const output_type& v, async_node_type::gateway_type& gateway) { 
		/* consume output data from async work */
		gateway.reserve_wait();
		printf("workConsumer: received value of %d\n", v);		// works better than using cout, since prints the entire string in a single shot, since multi-threaded
		gateway.release_wait();
	});

#if 0	// one way to get results
	tbb::flow::make_edge(asyncWorker, workConsumer);
	tbb::flow::make_edge(source, asyncWorker);		// connect source last because it starts generating immediately, while graph is being constructed, unless setup as disabled by default

	g.wait_for_all();	// complete all graph computation
#endif
#if 1	// another way to get results
	queue_node<output_type> result_queue(g);
	make_edge(output_port<0>(asyncWorker), result_queue);
	tbb::flow::make_edge(source, asyncWorker);		// connect source last because it starts generating immediately, while graph is being constructed, unless setup as disabled by default

	g.wait_for_all();	// complete all graph computation

	// Collect output
	output_type result;
	while (result_queue.try_get(result))
		printf("result_queue: received value of %d\n", (int)result);
#endif
	return 0;
}

#endif