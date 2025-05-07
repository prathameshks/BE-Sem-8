#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

using namespace std;


void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> leftPart(arr.begin() + left, arr.begin() + mid + 1);
    vector<int> rightPart(arr.begin() + mid + 1, arr.begin() + right + 1);

    int i = 0, j = 0, k = left;
    while (i < leftPart.size() && j < rightPart.size()) {
        if (leftPart[i] <= rightPart[j])
            arr[k++] = leftPart[i++];
        else
            arr[k++] = rightPart[j++];
    }

    while (i < leftPart.size()) arr[k++] = leftPart[i++];
    while (j < rightPart.size()) arr[k++] = rightPart[j++];
}

void sequentialMergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = (left + right) / 2;
        sequentialMergeSort(arr, left, mid);
        sequentialMergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void parallelMergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = (left + right) / 2;

        if (abs(mid - left) >= 100) {  // if array is large enough(atleast 100 elements), use parallelism
            #pragma omp parallel sections
            {
                #pragma omp section
                parallelMergeSort(arr, left, mid);
                #pragma omp section
                parallelMergeSort(arr, mid + 1, right);
            }
        } else { // for smaller array use sequential 
            sequentialMergeSort(arr, left, mid);
            sequentialMergeSort(arr, mid + 1, right);
        }

        merge(arr, left, mid, right);
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
    double start,end;
    
    start = omp_get_wtime();
    sequentialMergeSort(a1, 0, N - 1);
    end = omp_get_wtime();
    cout << "Sequential Merge Sort:  " << end - start<< " sec\n";

    start = omp_get_wtime();
    parallelMergeSort(a2, 0, N - 1);
    end = omp_get_wtime();
    cout << "Parallel Merge Sort:    " << end - start<< " sec\n";

    return 0;
}

// Compile with: g++ -fopenmp hpc2_merge_sort.cpp
// Run with: ./a.out or ./a.exe

// input:
// 15000

// 15000- number of elements