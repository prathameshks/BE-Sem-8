#include <omp.h>

#include <iostream>
#include <queue>

using namespace std;

class Node {
   public:
    int data;
    Node* left;
    Node* right;
    Node(int val) {
        data = val;
        left = NULL;
        right = NULL;
    }

    // generate a tree from array of integers given as BFS representation
    static Node* generate_tree(int* arr, int n) {
        if (n == 0) return NULL;
        Node* root = new Node(arr[0]);
        Node* current;
        queue<Node*> q;
        int i = 1;

        q.push(root);

        while (!q.empty() && i < n) {
            current = q.front();
            q.pop();

            current->left = new Node(arr[i++]);
            q.push(current->left);

            if (i >= n) break;

            current->right = new Node(arr[i++]);
            q.push(current->right);
        }

        return root;
    }
};

void bfs(Node* root) {
    if (root == NULL) {
        return;
    }
    queue<Node*> q;
    q.push(root);
    Node* temp;
    while (!q.empty()) {
        temp = q.front();
        cout << temp->data << " -> ";
        if (temp->left != NULL) {
            q.push(temp->left);
        }
        if (temp->right != NULL) {
            q.push(temp->right);
        }
        q.pop();
    }
}

void dfs(Node* root) {
    if (root == NULL) {
        return;
    }

    cout << root->data << " -> ";

    dfs(root->left);
    dfs(root->right);
}



// Parallel DFS using OpenMP
void dfs_parallel(Node* root) {
    if (root == NULL) {
        return;
    }

#pragma omp critical
    cout << root->data << " -> ";

#pragma omp parallel sections num_threads(2)

    {
#pragma omp section
        {
            dfs_parallel(root->left);
        }

#pragma omp section
        {
            dfs_parallel(root->right);
        }
    }
}

int main() {
    cout << "Start of BFS/DFS OpenMP program" << endl;
    int size = 100;
    int arr[100];
    for (int i = 0; i < size; i++) {
        arr[i] = i + 1;
    }
    Node* root = Node::generate_tree(arr, size);

    cout << "Tree generated from array of integers" << endl;

    double start_time_bfs = omp_get_wtime();
    bfs(root);
    double end_time_bfs = omp_get_wtime();

    double bfs_time = end_time_bfs - start_time_bfs;
    cout << endl << "TIME TAKEN BFS:" << bfs_time << endl;

    double start_time_dfs = omp_get_wtime();
    dfs(root);
    double end_time_dfs = omp_get_wtime();
    double dfs_time = end_time_dfs - start_time_dfs;
    cout << endl << "TIME TAKEN DFS:" << dfs_time << endl;

    double start_time_dfs_parallel = omp_get_wtime();
    dfs_parallel(root);
    double end_time_dfs_parallel = omp_get_wtime();
    double dfs_parallel_time = end_time_dfs_parallel - start_time_dfs_parallel;
    cout << endl << "TIME TAKEN DFS PARALLEL:" << dfs_parallel_time << endl;

    return 0;
}
