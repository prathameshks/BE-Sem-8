#include <omp.h>
#include <iostream>
#include <vector>

using namespace std;


vector<vector<int>> adj;  // adjacency list
vector<bool>  visited;     // mark visited nodes

void dfs(int node) {
    if (visited[node]) return;  // if already visited, return
    cout << node << " -> ";  // print the current node
    visited[node] = true;
#pragma omp parallel for
    for (int i = 0; i < adj[node].size(); i++) {
        int next_node = adj[node][i];
        if (!visited[next_node]) {
            dfs(next_node);
        }
    }
}

int main() {
    cout << "Please enter nodes:";
    int n, m;  // number of nodes and edges
    cin >> n;
    cout << "Please enter edges:";
    cin >> m;

    adj.resize(n + 5);  // resize adjacency list
    visited.resize(n + 5, false);  // resize and initialize visited vector

    for (int i = 1; i <= m; i++) {
        int u, v;  // edge between u and v
        u = rand() % n + 1;  // random node u
        v = rand() % n + 1;  // random node v
        // cout << u << " " << v << endl;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    int start_node;  // start node of DFS
    cout << "Please enter start node:";
    cin >> start_node;
    
    double start_time = omp_get_wtime();  // Ensure OpenMP is linked during compilation
    dfs(start_node);
    double end_time = omp_get_wtime();  // Ensure OpenMP is linked during compilation
    cout << "Time taken for DFS: " << end_time - start_time << " seconds" << endl;
    
    cout<<"Number of threads used: " << omp_get_max_threads() << endl;

    return 0;
}

// Compile with: g++ -fopenmp hpc1_dfs.cpp
// Run with: ./a.out or ./a.exe

// input:
// 20 50 2

// 20- number of vertices
// 50- number of edges
// 2- start node