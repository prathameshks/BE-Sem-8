#include <omp.h>

#include <iostream>
#include <queue>
#include <vector>
using namespace std;

// Parallel BFS using OpenMP
void bfs(int start, vector<vector<int>>& adj_list, vector<bool>& visited) {
    queue<int> q;
    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int curr_vertex = q.front();
        q.pop();
        cout << curr_vertex << " ";
#pragma omp parallel for shared(adj_list, visited, q) schedule(dynamic)
        for (int i = 0; i < adj_list[curr_vertex].size(); i++) {
            int neighbour = adj_list[curr_vertex][i];
            if (!visited[neighbour]) {
                visited[neighbour] = true;
                q.push(neighbour);
            }
        }
    }
}

int main() {
    int num_vertices, num_edges, source;
    cout << "Please enter number of vertices, edges and source node:" << endl;
    cin >> num_vertices >> num_edges >> source;
    vector<vector<int>> adj_list(num_vertices + 1);

    for (int i = 0; i < num_edges; i++) {
        int u, v;
        u = rand() % num_vertices + 1;  // random node u
        v = rand() % num_vertices + 1;  // random node v
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }
    vector<bool> visited(num_vertices + 1, false);

    double start_time = omp_get_wtime();
    bfs(source, adj_list, visited);
    double end_time = omp_get_wtime();

    cout << "\nTime taken for BFS: " << end_time - start_time << " seconds"
         << endl;
    cout << "Number of threads used: " << omp_get_max_threads() << endl;
    return 0;
}

// Compile with: g++ -fopenmp hpc1_bfs.cpp
// Run with: ./a.out or ./a.exe

// input:
// 20 50 2

// 20- number of vertices
// 50- number of edges
// 2- source node