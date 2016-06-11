#include <iostream>
#include <fstream>
#include <chrono>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <cstdlib>
#include <stdbool.h>
#include <assert.h>

using namespace std;
using namespace std::chrono;

static bool **adj;

high_resolution_clock::time_point take_timestamp() {
	return high_resolution_clock::now();
}

long long take_timestamp_dif(const high_resolution_clock::time_point& timestamp) {
	high_resolution_clock::time_point timestamp_2 = high_resolution_clock::now();
	return duration_cast<microseconds>(timestamp_2 - timestamp).count();
}

static inline __m128i XMMLOAD(void const *ptr) {
	return _mm_loadu_si128((__m128i const *)ptr);
}

static inline __m128i XMMLOAD_ALIGNED(void const *ptr) {
	return _mm_load_si128((__m128i const *)ptr);
}

static inline __m128i _mm_setone_epi8(void) {
	__m128i x = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
	return _mm_xor_si128(_mm_add_epi8(x, x), x);
}

static inline void sse_NOT(const bool *const a, const bool *const dest, const int N) {
	assert(N % 16 == 0);
	
	const __m128i* ptr = (__m128i*)a;
	const __m128i* ptr_end = (__m128i*)(a + N);
	__m128i* dst_ptr = (__m128i*)dest;

	__m128i vector_of_ones = _mm_setone_epi8();

	while(ptr < ptr_end) {
		__m128i result = _mm_xor_si128(XMMLOAD_ALIGNED(ptr), vector_of_ones);
		_mm_storeu_si128(dst_ptr, result);
		ptr++;
		dst_ptr++;
	}
}

static inline void sse_AND(const bool *const a, const bool *const b, const bool *const dest, const int N) {
	assert(N % 16 == 0);

	const __m128i* ptr = (__m128i*)a;
	const __m128i* ptr_end = (__m128i*)(a + N);
	const __m128i* ptr2 = (__m128i*)b;
	__m128i* dst_ptr = (__m128i*)dest;

	while (ptr < ptr_end) {
		__m128i result = _mm_and_si128(XMMLOAD_ALIGNED(ptr), XMMLOAD_ALIGNED(ptr2));
		_mm_storeu_si128(dst_ptr, result);
		ptr++;
		ptr2++;
		dst_ptr++;
	}
}

static inline void sse_OR(const bool *const a, const bool *const b, const bool *const dest, const int N) {
	assert(N % 16 == 0);

	const __m128i* ptr = (__m128i*)a;
	const __m128i* ptr_end = (__m128i*)(a + N);
	const __m128i* ptr2 = (__m128i*)b;
	__m128i* dst_ptr = (__m128i*)dest;

	while (ptr < ptr_end) {
		__m128i result = _mm_or_si128(XMMLOAD_ALIGNED(ptr), XMMLOAD_ALIGNED(ptr2));
		_mm_storeu_si128(dst_ptr, result);
		ptr++;
		ptr2++;
		dst_ptr++;
	}
}

static inline void sse_CLEAR(bool *const buffer, const int N) {
	__m128i zero = _mm_setzero_si128();

	for (__m128i* start = reinterpret_cast<__m128i*>(buffer),
		*end = reinterpret_cast<__m128i*>(&buffer[N]);
		start < end; start++) {
		_mm_store_si128(start, zero);
	}
}

static inline void clearWithMemset(bool *const buffer, const int N) {
	memset(buffer, 0, N);
}

static inline void sse_SETVAL(int* buffer, const bool *const mask, const int N, const int val) {
	assert(N % 16 == 0);

	const __m128i value = _mm_set1_epi32(val);
	const __m128i* buffer_ptr = (__m128i*)buffer;
	const __m128i* buffer_end_ptr = (__m128i*)(buffer + N);
	int* output_buffer_ptr = buffer;
	const bool* mask_ptr = mask;

	while(buffer_ptr < buffer_end_ptr) {
		__m128i mask_value = XMMLOAD(mask_ptr);
		__m128i unpacked_mask = _mm_unpacklo_epi16(_mm_unpacklo_epi8(mask_value, _mm_setzero_si128()), _mm_setzero_si128());
		__m128i buffer_mask = _mm_or_si128(_mm_slli_epi32(unpacked_mask, 31), _mm_srli_epi32(unpacked_mask, 31));

		__m128i result = _mm_add_epi32(value, XMMLOAD_ALIGNED(buffer_ptr));
		_mm_maskstore_epi32((output_buffer_ptr), buffer_mask, result);

		buffer_ptr++;
		output_buffer_ptr += 4;
		mask_ptr += 4;
	}
}

static inline bool sse_IS_ZERO(const bool *const buffer, const int N) {
	assert(N % 16 == 0);

	const __m128i zeros = _mm_setzero_si128();
	const __m128i* buffer_ptr = (__m128i*)buffer;
	const __m128i* buffer_end_ptr = (__m128i*)(buffer + N);

	while(buffer_ptr < buffer_end_ptr) {
		if (_mm_movemask_epi8(_mm_cmpeq_epi8(XMMLOAD_ALIGNED(buffer_ptr++), zeros)) != 0xffff) 
			return false;
	}
	return true;
}

static void run_bfs(bool *const visited, int* distance, const int N, const int roundedN, int source, 
	bool *const X, bool *const Y, bool *const Z, bool *const notvisited) {

	visited[source] = true;
	distance[source] = 0;

	X[source] = true;

	for (int level = 1; level < N; level++) {
		clearWithMemset(Y, roundedN);

		for (int j = 0; j < N; j++) {
			if (X[j] == true) {
				sse_NOT(visited, notvisited, roundedN);
				sse_AND(adj[j], notvisited, Z, roundedN);
				sse_OR(Y, Z, Y, roundedN);
			}
		}

		sse_OR(visited, Y, visited, roundedN);
		memcpy(X, Y, roundedN);
		sse_SETVAL(distance, Y, roundedN, level);

		if (sse_IS_ZERO(X, roundedN)) {
				break;
		}
	}
}

static inline int roundUpToSixteen(const int n) {
	return (n + 16 - 1) & ~(16 - 1);
}

int main(int argc, char *argv[]) {

	cout << "BFS Algorithm - SSE vectorized implementation" << endl;

	if (argc != 5) {
		cout << "Usage: " << argv[0] << " <filename> <start_vertex> <test_runs> <show_distances>" << endl;
		return 0;
	}
	
	const int START_VERTEX = atoi(argv[2]);
	const int TOTAL_RUNS = atoi(argv[3]);
	const int SHOW_DISTANCES = atoi(argv[4]);

	cout << "Reading graph representation from file: " << argv[1] << endl;

	ifstream fin;
	fin.open(argv[1], ios::in);

	unsigned N, Q;

	fin >> N >> Q;

	cout << "N = " << N << ", Q = " << Q << endl;

	const int roundedN = roundUpToSixteen(N);

	adj = new bool*[roundedN];
	if (adj == NULL) {
		cout << "Cannot allocated memory for graph representation matrix" << endl;
		return 1;
	}

	for (int i = 0; i < roundedN; i++) {
		adj[i] = (bool *)_aligned_malloc(roundedN * sizeof(bool), 16);

		if (adj[i] == NULL) {
			cout << "Cannot allocated memory for graph representation matrix" << endl;
			return 1;
		}

		clearWithMemset(adj[i], roundedN);
	}
	
	unsigned a, b;
	for (int i = 0; i < Q; i++) {
		fin >> a >> b;
		adj[a][b] = adj[b][a] = true;
	}

	bool* visited = static_cast<bool *>(_aligned_malloc(roundedN, 16));
	bool* negvisited = static_cast<bool *>(_aligned_malloc(roundedN, 16));
	int *distance = static_cast<int *>(_aligned_malloc(roundedN * sizeof(int), 16));
	bool *Y = static_cast<bool *>(_aligned_malloc(roundedN, 16));
	bool *X = static_cast<bool *>(_aligned_malloc(roundedN, 16));
	bool *Z = static_cast<bool *>(_aligned_malloc(roundedN, 16));

	long long *time_results = new long long[TOTAL_RUNS];

	cout << "Running " << TOTAL_RUNS << " BFS tests starting from vertex " << START_VERTEX << endl;

	long long res = 0;
	for (int run = 0; run < TOTAL_RUNS; run++) {
		clearWithMemset(visited, roundedN);
		memset(distance, 0, sizeof(int) * roundedN);
		clearWithMemset(X, roundedN);
		clearWithMemset(Z, roundedN);
		clearWithMemset(Y, roundedN);
		auto timestamp1 = take_timestamp();
		run_bfs(visited, distance, N, roundedN, START_VERTEX, X, Y, Z, negvisited);
		time_results[run] = take_timestamp_dif(timestamp1);
		res += time_results[run];
	}

	cout << "After " << TOTAL_RUNS << " runs; AVG = " << res / TOTAL_RUNS << " useconds" << endl;

	if (SHOW_DISTANCES) {
		cout << "Calculated distance table: " << endl;
		for (int i = 0; i < N; i++) {
			cout << distance[i] << " ";
		}
	}
	
	for(int i = 0; i < roundedN; i++) {
		_aligned_free(adj[i]);
	}
	
	delete[] time_results;

	delete[] adj;

	_aligned_free(visited);
	_aligned_free(negvisited);
	_aligned_free(distance);

	_aligned_free(Z);
	_aligned_free(Y);
	_aligned_free(X);
	
	return 0;
}
