#include <omp.h>

#include <climits>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void min_reduction(vector<int>& arr) {
    int min_value = INT_MAX;
#pragma omp parallel for reduction(min : min_value)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] < min_value) {
            min_value = arr[i];
        }
    }
    cout << "Minimum value: " << min_value << endl;
}

void max_reduction(vector<int>& arr) {
    int max_value = INT_MIN;
#pragma omp parallel for reduction(max : max_value)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] > max_value) {
            max_value = arr[i];
        }
    }
    cout << "Maximum value: " << max_value << endl;
}

void sum_reduction(vector<int>& arr) {
    int sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    cout << "Sum: " << sum << endl;
}

void average_reduction(vector<int>& arr) {
    int sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    cout << "Average: " << (double)sum / arr.size() << endl;
}

int main() {
    int size;
    cout << "Enter the size of the array: ";
    cin >> size;
    
    vector<int> arr(size);
    generate(arr.begin(), arr.end(), rand);

    cout << "Array: "<<endl;
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    min_reduction(arr);
    max_reduction(arr);
    sum_reduction(arr);
    average_reduction(arr);
}

// Compile with: g++ -fopenmp hpc3_parallel_reduction.cpp
// Run with: ./a.out or ./a.exe

// input:
// 1000
// 1000- number of elements in the array