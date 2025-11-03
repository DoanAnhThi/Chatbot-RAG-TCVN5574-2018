"""
Latency Metric for RAG System Evaluation

Latency measures the response time of the RAG system, which is crucial for
user experience and system performance monitoring. This includes:
- End-to-end latency (total response time)
- Retrieval latency (time to fetch relevant documents)
- Generation latency (time to generate answer)
- Tokenization/parsing latency

Range: Typically measured in milliseconds or seconds.
Lower latency indicates better performance.
"""

from typing import List, Dict, Union, Optional, Callable
import time
from functools import wraps
from contextlib import contextmanager
import statistics


class LatencyTracker:
    """Tracks latency metrics for RAG system components."""

    def __init__(self):
        self.measurements: Dict[str, List[float]] = {}
        self.current_timers: Dict[str, float] = {}

    def start_timer(self, name: str) -> None:
        """
        Start timing for a component.

        Args:
            name: Name of the component/operation
        """
        self.current_timers[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """
        Stop timing and record the measurement.

        Args:
            name: Name of the component/operation

        Returns:
            Elapsed time in seconds
        """
        if name not in self.current_timers:
            raise ValueError(f"Timer '{name}' was not started")

        elapsed = time.time() - self.current_timers[name]
        del self.current_timers[name]

        if name not in self.measurements:
            self.measurements[name] = []

        self.measurements[name].append(elapsed)
        return elapsed

    @contextmanager
    def timer(self, name: str):
        """
        Context manager for timing operations.

        Args:
            name: Name of the component/operation
        """
        self.start_timer(name)
        try:
            yield
        finally:
            self.stop_timer(name)

    def record_measurement(self, name: str, duration: float) -> None:
        """
        Manually record a latency measurement.

        Args:
            name: Name of the component/operation
            duration: Duration in seconds
        """
        if name not in self.measurements:
            self.measurements[name] = []
        self.measurements[name].append(duration)

    def get_statistics(self, name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get latency statistics.

        Args:
            name: Specific component name, or None for all

        Returns:
            Dictionary with latency statistics
        """
        if name:
            measurements = self.measurements.get(name, [])
            if not measurements:
                return {}
        else:
            measurements = {}
            for comp_name, comp_measurements in self.measurements.items():
                measurements[comp_name] = comp_measurements

        if name:
            # Single component statistics
            stats = self._calculate_stats(measurements)
            return {name: stats}
        else:
            # All components statistics
            all_stats = {}
            for comp_name, comp_measurements in measurements.items():
                all_stats[comp_name] = self._calculate_stats(comp_measurements)
            return all_stats

    def _calculate_stats(self, measurements: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of measurements."""
        if not measurements:
            return {}

        return {
            'count': len(measurements),
            'mean': statistics.mean(measurements),
            'median': statistics.median(measurements),
            'min': min(measurements),
            'max': max(measurements),
            'std_dev': statistics.stdev(measurements) if len(measurements) > 1 else 0.0,
            'p95': self._percentile(measurements, 95),
            'p99': self._percentile(measurements, 99)
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile from data."""
        if not data:
            return 0.0
        data_sorted = sorted(data)
        k = (len(data_sorted) - 1) * (percentile / 100.0)
        f = int(k)
        c = k - f
        if f + 1 < len(data_sorted):
            return data_sorted[f] * (1 - c) + data_sorted[f + 1] * c
        else:
            return data_sorted[f]

    def reset(self) -> None:
        """Reset all measurements."""
        self.measurements.clear()
        self.current_timers.clear()


def measure_latency(func: Callable) -> Callable:
    """
    Decorator to measure function latency.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracker = getattr(wrapper, '_tracker', LatencyTracker())
        wrapper._tracker = tracker

        func_name = f"{func.__module__}.{func.__name__}" if func.__module__ else func.__name__

        with tracker.timer(func_name):
            result = func(*args, **kwargs)

        return result

    return wrapper


class RAGLatencyProfiler:
    """Profiler for RAG system latency measurement."""

    def __init__(self):
        self.tracker = LatencyTracker()

    def profile_retrieval(self, retrieval_func: Callable, *args, **kwargs) -> tuple:
        """
        Profile retrieval latency.

        Args:
            retrieval_func: Retrieval function to profile
            *args, **kwargs: Arguments for the retrieval function

        Returns:
            Tuple of (result, latency_stats)
        """
        with self.tracker.timer('retrieval'):
            result = retrieval_func(*args, **kwargs)

        return result, self.tracker.get_statistics('retrieval')

    def profile_generation(self, generation_func: Callable, *args, **kwargs) -> tuple:
        """
        Profile generation latency.

        Args:
            generation_func: Generation function to profile
            *args, **kwargs: Arguments for the generation function

        Returns:
            Tuple of (result, latency_stats)
        """
        with self.tracker.timer('generation'):
            result = generation_func(*args, **kwargs)

        return result, self.tracker.get_statistics('generation')

    def profile_end_to_end(self, rag_func: Callable, *args, **kwargs) -> tuple:
        """
        Profile end-to-end RAG latency.

        Args:
            rag_func: Complete RAG function to profile
            *args, **kwargs: Arguments for the RAG function

        Returns:
            Tuple of (result, latency_stats)
        """
        with self.tracker.timer('end_to_end'):
            result = rag_func(*args, **kwargs)

        return result, self.tracker.get_statistics('end_to_end')

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all latency statistics."""
        return self.tracker.get_statistics()

    def reset(self) -> None:
        """Reset the profiler."""
        self.tracker.reset()


def calculate_average_latency(latencies: List[float]) -> Dict[str, float]:
    """
    Calculate average latency statistics.

    Args:
        latencies: List of latency measurements in seconds

    Returns:
        Dictionary with latency statistics
    """
    if not latencies:
        return {}

    return {
        'average_latency_ms': statistics.mean(latencies) * 1000,
        'median_latency_ms': statistics.median(latencies) * 1000,
        'min_latency_ms': min(latencies) * 1000,
        'max_latency_ms': max(latencies) * 1000,
        'p95_latency_ms': _percentile(latencies, 95) * 1000,
        'p99_latency_ms': _percentile(latencies, 99) * 1000
    }


def _percentile(data: List[float], percentile: float) -> float:
    """Calculate percentile from data."""
    if not data:
        return 0.0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (percentile / 100.0)
    f = int(k)
    c = k - f
    if f + 1 < len(data_sorted):
        return data_sorted[f] * (1 - c) + data_sorted[f + 1] * c
    else:
        return data_sorted[f]


# Example usage
if __name__ == "__main__":
    import time

    # Create latency tracker
    tracker = LatencyTracker()

    # Example 1: Manual timing
    print("Example 1: Manual timing")
    tracker.start_timer('example_operation')

    # Simulate some work
    time.sleep(0.1)

    latency = tracker.stop_timer('example_operation')
    print(".4f")

    # Example 2: Context manager
    print("\nExample 2: Context manager")
    with tracker.timer('another_operation'):
        time.sleep(0.05)

    # Get statistics
    stats = tracker.get_statistics()
    print("Statistics:")
    for component, component_stats in stats.items():
        print(f"  {component}: {component_stats}")

    # Example 3: RAG profiler
    print("\nExample 3: RAG profiler")

    def mock_retrieval(query):
        time.sleep(0.02)  # Simulate retrieval time
        return ["doc1", "doc2", "doc3"]

    def mock_generation(contexts):
        time.sleep(0.08)  # Simulate generation time
        return "This is a generated answer based on the retrieved documents."

    def mock_rag_pipeline(query):
        docs = mock_retrieval(query)
        answer = mock_generation(docs)
        return answer

    profiler = RAGLatencyProfiler()

    # Profile components
    _, retrieval_stats = profiler.profile_retrieval(mock_retrieval, "What is AI?")
    _, generation_stats = profiler.profile_generation(mock_generation, ["doc1", "doc2"])
    _, e2e_stats = profiler.profile_end_to_end(mock_rag_pipeline, "What is AI?")

    print("Retrieval latency stats:", retrieval_stats.get('retrieval', {}))
    print("Generation latency stats:", generation_stats.get('generation', {}))
    print("End-to-end latency stats:", e2e_stats.get('end_to_end', {}))

    # Example 4: Decorator
    print("\nExample 4: Function decorator")

    @measure_latency
    def example_function():
        time.sleep(0.03)
        return "Done"

    # Call the function multiple times
    for _ in range(3):
        result = example_function()

    # Get decorator statistics
    decorator_stats = example_function._tracker.get_statistics()
    print("Decorator statistics:", decorator_stats)
