#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

struct DataPoint {
    vector<float> data;  // Feature vector of the data point
    vector<float> context;  // Context vector (e.g., time, location, etc.)

    DataPoint(vector<float> d, vector<float> c) : data(d), context(c) {}
};

class ContextualClusterExpansion {
private:
    float contextWeight;
    float distanceThreshold;
    int maxIterations;
    vector<vector<DataPoint>> clusters;
    vector<vector<float>> contextualVectors;

public:
    ContextualClusterExpansion(float cWeight = 0.5, float distThreshold = 1.0, int maxIter = 100)
        : contextWeight(cWeight), distanceThreshold(distThreshold), maxIterations(maxIter) {}

    float calculateSimilarity(const DataPoint& point, const vector<float>& clusterCenter, const vector<float>& contextVector) {
        // Calculate Euclidean distance + context similarity
        float featureSimilarity = 0.0;
        for (size_t i = 0; i < point.data.size(); ++i) {
            featureSimilarity += pow(point.data[i] - clusterCenter[i], 2);
        }
        featureSimilarity = sqrt(featureSimilarity);

        float contextSimilarity = 0.0;
        for (size_t i = 0; i < point.context.size(); ++i) {
            contextSimilarity += point.context[i] * contextVector[i];
        }

        return featureSimilarity * (1 - contextWeight) + contextSimilarity * contextWeight;
    }

    void addPointToCluster(const DataPoint& point, int clusterIdx) {
        clusters[clusterIdx].push_back(point);
        contextualVectors[clusterIdx] = calculateClusterContext(clusterIdx);
    }

    vector<float> calculateClusterContext(int clusterIdx) {
        // Calculate the average context vector of the cluster
        vector<float> avgContext(contextualVectors[clusterIdx].size(), 0.0);
        for (const auto& point : clusters[clusterIdx]) {
            for (size_t i = 0; i < point.context.size(); ++i) {
                avgContext[i] += point.context[i];
            }
        }
        for (size_t i = 0; i < avgContext.size(); ++i) {
            avgContext[i] /= clusters[clusterIdx].size();
        }
        return avgContext;
    }

    void createNewCluster(const DataPoint& point) {
        clusters.push_back({point});
        contextualVectors.push_back(point.context);
    }

    void fit(const vector<DataPoint>& dataStream) {
        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            for (const auto& point : dataStream) {
                int bestClusterIdx = -1;
                float bestSimilarity = FLT_MAX;

                for (size_t i = 0; i < clusters.size(); ++i) {
                    // Calculate the center of the cluster (mean of the features)
                    vector<float> clusterCenter(point.data.size(), 0.0);
                    for (const auto& p : clusters[i]) {
                        for (size_t j = 0; j < p.data.size(); ++j) {
                            clusterCenter[j] += p.data[j];
                        }
                    }
                    for (size_t j = 0; j < clusterCenter.size(); ++j) {
                        clusterCenter[j] /= clusters[i].size();
                    }

                    // Calculate similarity between point and the cluster center, considering context
                    float similarity = calculateSimilarity(point, clusterCenter, contextualVectors[i]);

                    if (similarity < bestSimilarity && similarity < distanceThreshold) {
                        bestSimilarity = similarity;
                        bestClusterIdx = i;
                    }
                }

                if (bestClusterIdx == -1) {
                    createNewCluster(point);
                } else {
                    addPointToCluster(point, bestClusterIdx);
                }
            }

            // Optional: Adjust clusters based on evolving context or features
            // This step can be expanded to include merging or splitting clusters if needed
        }
    }

    int predict(const DataPoint& newData) {
        int bestClusterIdx = -1;
        float bestSimilarity = FLT_MAX;

        for (size_t i = 0; i < clusters.size(); ++i) {
            // Calculate the center of the cluster (mean of the features)
            vector<float> clusterCenter(newData.data.size(), 0.0);
            for (const auto& p : clusters[i]) {
                for (size_t j = 0; j < p.data.size(); ++j) {
                    clusterCenter[j] += p.data[j];
                }
            }
            for (size_t j = 0; j < clusterCenter.size(); ++j) {
                clusterCenter[j] /= clusters[i].size();
            }

            // Calculate similarity between the new data point and the cluster center, considering context
            float similarity = calculateSimilarity(newData, clusterCenter, contextualVectors[i]);

            if (similarity < bestSimilarity) {
                bestSimilarity = similarity;
                bestClusterIdx = i;
            }
        }
        return bestClusterIdx;
    }

    void printClusters() {
        for (size_t i = 0; i < clusters.size(); ++i) {
            cout << "Cluster " << i + 1 << " (" << clusters[i].size() << " points):\n";
            for (const auto& point : clusters[i]) {
                cout << "Data: [";
                for (const auto& value : point.data) {
                    cout << value << " ";
                }
                cout << "] Context: [";
                for (const auto& value : point.context) {
                    cout << value << " ";
                }
                cout << "]\n";
            }
            cout << endl;
        }
    }
};

int main() {
    // Example usage with some toy data points (feature vector + context)
    vector<DataPoint> dataStream = {
        DataPoint({1.0, 2.0}, {0.5, 1.0}),
        DataPoint({1.5, 2.5}, {0.6, 1.1}),
        DataPoint({8.0, 8.5}, {2.0, 2.5}),
        DataPoint({7.5, 8.0}, {2.1, 2.6}),
        DataPoint({1.2, 2.3}, {0.7, 1.2}),
    };

    // Initialize the Contextual Cluster Expansion algorithm
    ContextualClusterExpansion cce;

    // Fit the model to the data stream
    cce.fit(dataStream);

    // Print the clusters
    cce.printClusters();

    // Predict the cluster of a new data point
    DataPoint newPoint({1.1, 2.2}, {0.5, 1.0});
    int predictedCluster = cce.predict(newPoint);
    cout << "New point predicted to belong to cluster " << predictedCluster + 1 << endl;

    return 0;
}
