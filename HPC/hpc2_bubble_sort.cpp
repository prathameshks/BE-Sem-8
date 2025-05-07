#include <omp.h>

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1]) swap(arr[j], arr[j + 1]);
}

void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) { // for i 0 2 4 6 ... even
#pragma omp parallel for
            for (int j = 0; j < n - 1; j += 2) {
                if (arr[j] > arr[j + 1]) swap(arr[j], arr[j + 1]);
            }
        } else { // for i 1 3 5 7 ... odd
#pragma omp parallel for
            for (int j = 1; j < n - 1; j += 2) {
                if (arr[j] > arr[j + 1]) swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    int N;
    cout << "Please enter number of elements: ";
    cin >> N;
    if (N <= 0) {
        cout << "Invalid input. Exiting." << endl;
        return 1;
    }
    vector<int> original(N);
    generate(original.begin(), original.end(), rand);
    vector<int> a1(N);
    copy(original.begin(), original.end(), a1.begin());
    vector<int> a2(N);
    copy(original.begin(), original.end(), a2.begin());
    double start, end;

    start = omp_get_wtime();
    sequentialBubbleSort(a1);
    end = omp_get_wtime();
    cout << "Sequential Bubble Sort: " << end - start << " sec\n";

    a1 = original;
    start = omp_get_wtime();
    parallelBubbleSort(a2);
    end = omp_get_wtime();
    cout << "Parallel Bubble Sort:   " << end - start << " sec\n";

    return 0;
}

// Compile with: g++ -fopenmp hpc2_bubble_sort.cpp
// Run with: ./a.out or ./a.exe

// input: 
// 15000

// 15000- number of elements
// output: Sequential Bubble Sort: 0.123 sec
//         Parallel Bubble Sort:   0.045 sec