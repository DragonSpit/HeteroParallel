/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Example of integrating CUDA functions into an existing
 * application / framework.
 * CPP code representing the existing application / framework.
 * Compiled with default CPP compiler.
 */

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

//https://software.intel.com/en-us/videos/cpus-gpus-fpgas-managing-the-alphabet-soup-with-intel-threading-building-blocks

using namespace concurrency;
using namespace std;

using namespace tbb::flow;

// Tasks:
// 1. (Done) Integrate Intel file compression example into this project to see if it would build successfully.
// 2. (80% Done) Understand async_node file reader, compressor, and writer example completely.
// 3. (Done) Contact Intel for async_node Mandelbrot rendering example that load balances between CPU and GPU.
// 4. Until Intel provides Mandlebrot rendering example, work on adapting file compression example to our needs,
//    working first on understanding the example thoroughly.
// 5. Develop a design for how CPU and GPU will get work items of number of random numbers (float? unsigne?) to generate
// 6. 
//-----------------------------------------------------------------------------------------------------------------------
//---------------------------------------Compression example based on async_node-----------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// BufferMsg class contains inputBuffer and outputBuffer data members as well as sequence ID, along with an indicators of the last message
struct Buffer {
	size_t len;
	char* b;
};

struct BufferMsg {

	BufferMsg() {}
	BufferMsg(Buffer& inputBuffer, Buffer& outputBuffer, size_t seqId, bool isLast = false)
		: inputBuffer(inputBuffer), outputBuffer(outputBuffer), seqId(seqId), isLast(isLast) {}

	static BufferMsg createBufferMsg(size_t seqId, size_t chunkSize) {
		Buffer inputBuffer;
		inputBuffer.b = new char[chunkSize];
		inputBuffer.len = chunkSize;

		Buffer outputBuffer;
		size_t compressedChunkSize = (size_t)(chunkSize * 1.01 + 600); // compression overhead
		outputBuffer.b = new char[compressedChunkSize];
		outputBuffer.len = compressedChunkSize;

		return BufferMsg(inputBuffer, outputBuffer, seqId);
	}

	static void destroyBufferMsg(const BufferMsg& destroyMsg) {
		delete[] destroyMsg.inputBuffer.b;
		delete[] destroyMsg.outputBuffer.b;
	}

	void markLast(size_t lastId) {
		isLast = true;
		seqId = lastId;
	}

	size_t seqId;
	Buffer inputBuffer;
	Buffer outputBuffer;
	bool isLast;
};

// VjD: Not part of the example, but my extension of BufferMsg structure to be more general
struct BufferIOMsg {

	BufferIOMsg() {}
	BufferIOMsg(Buffer& inputBuffer, Buffer& outputBuffer, size_t seqId, bool isLast = false)
		: inputBuffer(inputBuffer), outputBuffer(outputBuffer), seqId(seqId), isLast(isLast) {}

	static BufferIOMsg createBufferIOMsg(size_t seqId, size_t inChunkSize, size_t outChunkSize) {
		Buffer inputBuffer;
		inputBuffer.b = new char[inChunkSize];
		inputBuffer.len = inChunkSize;

		Buffer outputBuffer;
		outputBuffer.b = new char[outChunkSize];
		outputBuffer.len = outChunkSize;

		return BufferIOMsg(inputBuffer, outputBuffer, seqId);
	}

	static void destroyBufferMsg(const BufferMsg& destroyMsg) {
		delete[] destroyMsg.inputBuffer.b;
		delete[] destroyMsg.outputBuffer.b;
	}

	void markLast(size_t lastId) {
		isLast = true;
		seqId = lastId;
	}

	size_t seqId;
	Buffer inputBuffer;
	Buffer outputBuffer;
	bool isLast;
};

// Used by the compressor function_node, which calls the () method to do the actual compression work
class BufferCompressor {
public:

	BufferCompressor(int blockSizeIn100KB) : m_blockSize(blockSizeIn100KB) {}

	// VjD Takes the input buffer, compresses it and returns the same buffer, since the buffer has an inputBuffer member and an outputBuffer member (in which the compressed
	// VjD output is placed
	BufferMsg operator()(BufferMsg buffer) const {
		if (!buffer.isLast) {
			unsigned int outSize = (unsigned int)buffer.outputBuffer.len;
			BZ2_bzBuffToBuffCompress(buffer.outputBuffer.b, &outSize, buffer.inputBuffer.b, (unsigned int)buffer.inputBuffer.len, m_blockSize, 0, 30);
			buffer.outputBuffer.len = outSize;
		}
		return buffer;
	}

private:
	int m_blockSize;
};

// Uses input stream to read from and put into a Buffer, or to write to an output stream from a Buffer.
// Chunk size of reads and writes stays constant, specified at construction time.
// Number of chunks that have been read can be queried
// If there is more data available in the input stream can be queried
class IOOperations {
public:

	IOOperations(std::ifstream& inputStream, std::ofstream& outputStream, size_t chunkSize)
		: m_inputStream(inputStream), m_outputStream(outputStream), m_chunkSize(chunkSize), m_chunksRead(0) {}

	void readChunk(Buffer& buffer) {
		m_inputStream.read(buffer.b, m_chunkSize);
		buffer.len = static_cast<size_t>(m_inputStream.gcount());
		m_chunksRead++;
	}

	void writeChunk(const Buffer& buffer) {
		m_outputStream.write(buffer.b, buffer.len);
	}

	size_t chunksRead() const {
		return m_chunksRead;
	}

	size_t chunkSize() const {
		return m_chunkSize;
	}

	bool hasDataToRead() const {
		return m_inputStream.is_open() && !m_inputStream.eof();
	}

private:

	std::ifstream& m_inputStream;
	std::ofstream& m_outputStream;

	size_t m_chunkSize;
	size_t m_chunksRead;
};

typedef tbb::flow::async_node< tbb::flow::continue_msg, BufferMsg > async_file_reader_node;
typedef tbb::flow::async_node< BufferMsg, tbb::flow::continue_msg > async_file_writer_node;

// Created with an IOOperation, which specifies the input and output streams
// Performs submitRead and submitWrite operations. submitRead waits for the gateway, creates a redingLoop thread and swaps it into the data member thread variable
// submitWrite copies a BufferMsg into a concurrent_bounded_queue
// fileWriterThread is created during construction and is initialized to writingLoop. It pops BufferMsg off the writeQueue and writes it into outputStream
// readingLoop creates a new BufferMsg, reads a chunk from inputStream and puts it into BufferMsg, sends it to the flowGraph gateway
// When there is no more data available from the inputSteam, readingLoop sends BufferMsg marksed last to the flowGraph gateway
class AsyncNodeActivity {
public:

	AsyncNodeActivity(IOOperations& io)
		: m_io(io), m_fileWriterThread(&AsyncNodeActivity::writingLoop, this) {}

	~AsyncNodeActivity() {
		m_fileReaderThread.join();
		m_fileWriterThread.join();
	}

	void submitRead(async_file_reader_node::gateway_type& gateway) {
		gateway.reserve_wait();
		std::thread(&AsyncNodeActivity::readingLoop, this, std::ref(gateway)).swap(m_fileReaderThread);
	}

	void submitWrite(const BufferMsg& bufferMsg) {
		m_writeQueue.push(bufferMsg);
	}

private:

	void readingLoop(async_file_reader_node::gateway_type& gateway) {
		while (m_io.hasDataToRead()) {
			BufferMsg bufferMsg = BufferMsg::createBufferMsg(m_io.chunksRead(), m_io.chunkSize());
			m_io.readChunk(bufferMsg.inputBuffer);
			gateway.try_put(bufferMsg);
		}
		sendLastMessage(gateway);
		gateway.release_wait();
	}

	void writingLoop() {
		BufferMsg buffer;
		m_writeQueue.pop(buffer);
		while (!buffer.isLast) {
			m_io.writeChunk(buffer.outputBuffer);
			m_writeQueue.pop(buffer);
		}
	}

	void sendLastMessage(async_file_reader_node::gateway_type& gateway) {
		BufferMsg lastMsg;
		lastMsg.markLast(m_io.chunksRead());
		gateway.try_put(lastMsg);
	}

	IOOperations& m_io;

	tbb::concurrent_bounded_queue< BufferMsg > m_writeQueue;

	std::thread m_fileReaderThread;
	std::thread m_fileWriterThread;
};

void fgCompressionAsyncNode(IOOperations& io, int blockSizeIn100KB)
{
	tbb::flow::graph g;

	AsyncNodeActivity asyncNodeActivity(io);

	async_file_reader_node file_reader(g, tbb::flow::unlimited, [&asyncNodeActivity](const tbb::flow::continue_msg& msg, async_file_reader_node::gateway_type& gateway) {
		asyncNodeActivity.submitRead(gateway);
	});

	tbb::flow::function_node< BufferMsg, BufferMsg > compressor(g, tbb::flow::unlimited, BufferCompressor(blockSizeIn100KB));

	// sequencer_node re-orders bufferMsg's based on (size_t bufferMsg.seqId)
	tbb::flow::sequencer_node< BufferMsg > ordering(g, [](const BufferMsg& bufferMsg)->size_t {
		return bufferMsg.seqId;
	});

	// The node is serial to preserve the right order of buffers set by the preceding sequencer_node
	async_file_writer_node output_writer(g, tbb::flow::serial, [&asyncNodeActivity](const BufferMsg& bufferMsg, async_file_writer_node::gateway_type& gateway) {
		asyncNodeActivity.submitWrite(bufferMsg);
	});

	make_edge(file_reader, compressor);					// create the graph
	make_edge(compressor, ordering);
	make_edge(ordering, output_writer);

	file_reader.try_put(tbb::flow::continue_msg());		// start operations up, from the front of the graph

	g.wait_for_all();
}

void runAsyncNode()
{
	const std::string archiveExtension = ".bz2";
	std::string inputFileName = "inputFile";
	int blockSizeIn100KB = 1; // block size in 100KB chunks

	std::ifstream inputStream(inputFileName.c_str(), std::ios::in | std::ios::binary);
	if (!inputStream.is_open()) {
		throw std::invalid_argument("Cannot open " + inputFileName + " file.");
	}

	std::string outputFileName(inputFileName + archiveExtension);

	std::ofstream outputStream(outputFileName.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
	if (!outputStream.is_open()) {
		throw std::invalid_argument("Cannot open " + outputFileName + " file.");
	}

	// General interface to work with I/O buffers operations
	size_t chunkSize = blockSizeIn100KB * 100 * 1024;
	IOOperations io(inputStream, outputStream, chunkSize);

	fgCompressionAsyncNode(io, blockSizeIn100KB);
	cout << "Ran AsyncNode file compression" << endl;
}

#endif