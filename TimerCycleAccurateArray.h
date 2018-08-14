// Copyright (c) Victor J. Duvanenko, 2009
// Abstraction for the CPU cycle accurate timer functions.
// Based on Intel's IPP functions that read CPU cycle counter and other info.
// Uses built-in arrays, since these provide more consistent performance and lower
// overhead than STL vector (which I also implemented and compared in performance).
// Measures the overhead of the measurement processes (i.e. calling the measurement
// and getting a time-stamp) and then subtracts that from all measurements to get
// a more accurate measurement, especially for very fast functions.
// Benchmarked QueryPerformanceCounter() windows function that also provides CPU
// clock cycles and it takes 495-441 clock cycles per call, versus IPP that takes 36-45 cycles.
// The difference does not really matter much, since both provide the information and
// the overhead of measurement is measured and then removed.  One possible advantage of using
// IPP method is that it provides a platform independent method (but requires the use of IPP library).

// In the future, it should be possible to timestamp around the memory re-allocation and then subtract this measured time
// from all subsequent timestamps.

#include <ostream>
#include <math.h>
#include "ipp.h"

#ifndef _TimerCPUcycleAccurateArray_h
#define _TimerCPUcycleAccurateArray_h

class TimerCycleAccurateArray {
public:
	 TimerCycleAccurateArray( unsigned long reserveSize = 100 );
	~TimerCycleAccurateArray();
	void    reset();						// clears all of the time-stamps
	void	timeStamp();					// the only command to add a time-stamp
	double	getElapsedTimeInSeconds();		// from the first time-stamp to the last
	__int64 getElapsedTimeInCycles();
	double	getLargestDeltaInSeconds();
	__int64	getLargestDeltaInCycles();
	double	getSmallestDeltaInSeconds();
	__int64	getSmallestDeltaInCycles();
	double	getAverageDeltaInSeconds();
	double	getAverageDeltaInCycles();
	__int64 getCycleCount(unsigned index);

	void	displayDeltas( std::ostream& os );

private:
	double			_cycleTime;
	__int64 *		_cycleCount;		// array of time-stamps
	unsigned long	_cycleCount_size;	// size  of the time-stamp array
	unsigned long	_nTimeStamps;		// number of time-stamps gathered so far
	__int64			_cyclesOverheadOfMeasurement;	// overhead of taking a measurement in CPU cycles.  Measured during construction.
};

inline TimerCycleAccurateArray::TimerCycleAccurateArray( unsigned long reserveSize )
{
	int CPUfrequencyInMHz;

	ippGetCpuFreqMhz( &CPUfrequencyInMHz );

	_cycleTime = 1.0 / ((double)CPUfrequencyInMHz * 1000000.0 );

	_cycleCount      = new __int64[ reserveSize ];
	_cycleCount_size = reserveSize;
	reset();

	// Establish a baseline measurement of how long it takes to do timestamping
	// and subtract it from all subsequent measurements
	_cyclesOverheadOfMeasurement = 0;		// must set this offset to zero, since it gets subtracted from all measurements
											// even in the average function below as we try to compute the value for _cyclesOverheadOfMeasurement
	timeStamp();	// Very imprortant that we measure exactly the function that will be used for measurement
	timeStamp();	// and all of the overhead within that function
	timeStamp();
	timeStamp();
	timeStamp();
	timeStamp();
	timeStamp();
	timeStamp();
	timeStamp();
	timeStamp();
	timeStamp();	// 11 time-stamps provide 10 deltas
	_cyclesOverheadOfMeasurement = (__int64)( getAverageDeltaInCycles() + 0.5 );

	reset();
}

inline TimerCycleAccurateArray::~TimerCycleAccurateArray()
{
	delete[] _cycleCount;
}

inline void TimerCycleAccurateArray::reset()
{
	_nTimeStamps = 0;
}
// Decided to not grow the array size dynamically, since that introduces delay when memory re-allocation is performed.
// In the future, it should be possible to timestamp around the memory re-allocation and then subtract this measured time
// from all subsequent timestamps.
inline void TimerCycleAccurateArray::timeStamp()
{
	if ( _nTimeStamps < ( _cycleCount_size - 1 ))			// make sure that we have enough room in the timestamp array
	{
		_cycleCount[ _nTimeStamps ] = ippGetCpuClocks();	// add a new time-stamp to the array
		_nTimeStamps++;
	}
}

inline __int64 TimerCycleAccurateArray::getElapsedTimeInCycles()
{
	return( _cycleCount[ _nTimeStamps - 1 ] - _cycleCount[ 0 ] );
}

inline __int64 TimerCycleAccurateArray::getCycleCount(unsigned index)
{
	return(_cycleCount[index]);
}

inline double TimerCycleAccurateArray::getElapsedTimeInSeconds()
{
	return( getElapsedTimeInCycles() * _cycleTime );
}

inline __int64 TimerCycleAccurateArray::getLargestDeltaInCycles()
{
	if ( _nTimeStamps <= 1 ) return 0;

	__int64	maxDelta = 0;
	for( unsigned long i = 1; i < _nTimeStamps; ++i )
	{
		if (( _cycleCount[ i ] - _cycleCount[ i - 1 ] ) > maxDelta )
			maxDelta = _cycleCount[ i ] - _cycleCount[ i - 1 ];
	}
	return( maxDelta - _cyclesOverheadOfMeasurement );
}
inline double TimerCycleAccurateArray::getLargestDeltaInSeconds()
{
	return( getLargestDeltaInCycles() * _cycleTime );
}

inline __int64 TimerCycleAccurateArray::getSmallestDeltaInCycles()
{
	if ( _nTimeStamps <= 1 ) return 0;

	__int64	minDelta = _cycleCount[ 1 ] - _cycleCount[ 0 ];
	for( unsigned long i = 2; i < _nTimeStamps; ++i )
	{
		if (( _cycleCount[ i ] - _cycleCount[ i - 1 ] ) < minDelta )
			minDelta = _cycleCount[ i ] - _cycleCount[ i - 1 ];
	}
	return( minDelta - _cyclesOverheadOfMeasurement );
}

inline double TimerCycleAccurateArray::getSmallestDeltaInSeconds()
{
	return( getSmallestDeltaInCycles() * _cycleTime );
}

inline double TimerCycleAccurateArray::getAverageDeltaInCycles()
{
	return( (double)getElapsedTimeInCycles() / ( _nTimeStamps - 1 ) - _cyclesOverheadOfMeasurement );	// there are 1 fewer deltas than timeStamps
}

inline double TimerCycleAccurateArray::getAverageDeltaInSeconds()
{
	return( (double)getElapsedTimeInSeconds() / ( _nTimeStamps - 1 ));	// there are 1 fewer deltas than timeStamps
}

inline void TimerCycleAccurateArray::displayDeltas( std::ostream& os )
{
	if ( _nTimeStamps <= 1 ) return;

	for( unsigned long i = 1; i < _nTimeStamps; ++i )
	{
		os << ( _cycleCount[ i ] - _cycleCount[ i - 1 ] - _cyclesOverheadOfMeasurement ) << "\t";
	}
	os << std::endl;
}


#endif