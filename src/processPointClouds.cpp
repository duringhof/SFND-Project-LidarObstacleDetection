// PCL lib Functions for processing point clouds

#include "processPointClouds.h"
#include <unordered_set>

//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}

//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}

template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(
    typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes,
    Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint) {

  // Time filtering process
  auto startTime = std::chrono::steady_clock::now();

  // Create the filtering object
  pcl::VoxelGrid<PointT> vg;
  typename pcl::PointCloud<PointT>::Ptr cloudFiltered (new pcl::PointCloud<PointT>);
  vg.setInputCloud (cloud);
  vg.setLeafSize (filterRes, filterRes, filterRes);
  vg.filter(*cloudFiltered);

  typename pcl::PointCloud<PointT>::Ptr cloudRegion(
      new pcl::PointCloud<PointT>);
  pcl::CropBox<PointT> region(true);
  region.setMin(minPoint);
  region.setMax(maxPoint);
  region.setInputCloud(cloudFiltered);
  region.filter(*cloudRegion);

  std::vector<int> indices;

  pcl::CropBox<PointT> roof(true);
  roof.setMin(Eigen::Vector4f(-1.5, -1.7, -1, 1));
  roof.setMax(Eigen::Vector4f(2.6, 1.7, 4, 1));
  roof.setInputCloud(cloudRegion);
  roof.filter(indices);

  pcl::PointIndices::Ptr inliers{new pcl::PointIndices};
  for (int point : indices)
    inliers->indices.push_back(point);

  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud(cloudRegion);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*cloudRegion);

  auto endTime = std::chrono::steady_clock::now();
  auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

  return cloudRegion;
}

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud)  {

  typename pcl::PointCloud<PointT>::Ptr obstCloud(
      new pcl::PointCloud<PointT>());
  typename pcl::PointCloud<PointT>::Ptr planeCloud(
      new pcl::PointCloud<PointT>());

  for (int index : inliers->indices)
    planeCloud->points.push_back(cloud->points[index]);

  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*obstCloud);

  std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstCloud, planeCloud);
  return segResult;
}

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold) {
    
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    pcl::PointIndices::Ptr inliers{new pcl::PointIndices};
    std::unordered_set<int> inliers_result;

    while (maxIterations--) {

      std::unordered_set<int> inliers_set;

      while (inliers_set.size() < 3)
        inliers_set.insert(rand() % (cloud->points.size()));

      float x1, y1, z1, x2, y2, z2, x3, y3, z3;
      auto itr = inliers_set.begin();
      x1 = cloud->points[*itr].x;
      y1 = cloud->points[*itr].y;
      z1 = cloud->points[*itr].z;
      itr++;
      x2 = cloud->points[*itr].x;
      y2 = cloud->points[*itr].y;
      z2 = cloud->points[*itr].z;
      itr++;
      x3 = cloud->points[*itr].x;
      y3 = cloud->points[*itr].y;
      z3 = cloud->points[*itr].z;

      float a = ((y2 - y1) * (z3 - z1)) - ((z2 - z1) * (y3 - y1));
      float b = ((z2 - z1) * (x3 - x1)) - ((x2 - x1) * (z3 - z1));
      float c = ((x2 - x1) * (y3 - y1)) - ((y2 - y1) * (x3 - x1));
      float d = -1 * (a * x1 + b * y1 + c * z1);

      for (int index = 0; index < cloud->points.size(); index++) {

        if (inliers_set.count(index) > 0)
          continue;

        PointT point = cloud->points[index];
        float x4 = point.x;
        float y4 = point.y;
        float z4 = point.z;

        float dist =
            fabs(a * x4 + b * y4 + c * z4 + d) / sqrt(a * a + b * b + c * c);

        if (dist <= distanceThreshold)
          inliers_set.insert(index);
      }

      if (inliers_set.size() > inliers_result.size())
        inliers_result = inliers_set;
    }

    for (auto i : inliers_result) {
      inliers->indices.push_back(i);
    }

    std::pair<typename pcl::PointCloud<PointT>::Ptr,
              typename pcl::PointCloud<PointT>::Ptr>
        segResult = SeparateClouds(inliers, cloud);
    
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    return segResult;
}

template <typename PointT>
void ProcessPointClouds<PointT>::proximity(
    int id, typename pcl::PointCloud<PointT>::Ptr cloud,
    std::vector<int> &cluster, std::vector<bool> &processed, KdTree *tree,
    float distanceTol) {

  processed[id] = true;
  cluster.push_back(id);

  std::vector<float> point = {cloud->points[id].x, cloud->points[id].y,
                              cloud->points[id].z};

  std::vector<int> nearest = tree->search(point, distanceTol);

  for (int point_id : nearest) {
    
    if (!processed[point_id]) {
      proximity(point_id, cloud, cluster, processed, tree, distanceTol);
    }
  }
}

template <typename PointT>
std::vector<std::vector<int>> ProcessPointClouds<PointT>::euclideanCluster(
    typename pcl::PointCloud<PointT>::Ptr cloud, KdTree *tree,
    float distanceTol) {

  std::vector<std::vector<int>> clusters;

  std::vector<bool> processed(cloud->points.size(), false);

  for(int i = 0; i < cloud->points.size(); i++) {

    if (!processed[i]) {

      std::vector<int> cluster;
      proximity(i, cloud, cluster, processed, tree, distanceTol);
      clusters.push_back(cluster);
    }
  }

  return clusters;
}

template <typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr>
ProcessPointClouds<PointT>::Clustering(
    typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance,
    int minSize, int maxSize) {

  // Time clustering process
  auto startTime = std::chrono::steady_clock::now();

  std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

  KdTree *tree = new KdTree;

  for (int i = 0; i < cloud->points.size(); i++) {
    std::vector<float> point = {cloud->points[i].x, cloud->points[i].y,
                                cloud->points[i].z};
    tree->insert(point, i);
  }

  std::vector<std::vector<int>> cluster_indices =
      euclideanCluster(cloud, tree, clusterTolerance);

  for (auto indices : cluster_indices) {

    typename pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>);
    for (int index : indices) {
      cluster->push_back(cloud->points[index]);
    }
    clusters.push_back(cluster);
  }

  auto endTime = std::chrono::steady_clock::now();
  auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

  return clusters;
}

template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::ClusteringPCL(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    typename pcl::search::KdTree<PointT>::Ptr tree(
        new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(clusterIndices);

    for (pcl::PointIndices getIndices : clusterIndices) {
      
      typename pcl::PointCloud<PointT>::Ptr cloudCluster(
          new pcl::PointCloud<PointT>);

      for (int index : getIndices.indices)
        cloudCluster->points.push_back(cloud->points[index]);

      cloudCluster->width = cloudCluster->points.size();
      cloudCluster->height = 1;
      cloudCluster->is_dense = true;

      clusters.push_back(cloudCluster);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}

template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}

template <typename PointT>
BoxQ ProcessPointClouds<PointT>::BoundingBoxQ(
    typename pcl::PointCloud<PointT>::Ptr cluster) {

  BoxQ box;
  // Compute principal directions
  Eigen::Vector4f pcaCentroid;
  pcl::compute3DCentroid(*cluster, pcaCentroid);
  Eigen::Matrix3f covariance;
  computeCovarianceMatrixNormalized(*cluster, pcaCentroid, covariance);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
  Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
  eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

  // Transform the original cloud to the origin where the principal components correspond to the axes.
  Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
  projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose();
  projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * pcaCentroid.head<3>());
  typename pcl::PointCloud<PointT>::Ptr cloudPointsProjected (new pcl::PointCloud<PointT>);
  pcl::transformPointCloud(*cluster, *cloudPointsProjected,
                           projectionTransform);
  
  // Get the minimum and maximum points of the transformed cloud.
  PointT minPoint, maxPoint;
  pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
  const Eigen::Vector3f meanDiagonal =
      0.5f * (maxPoint.getVector3fMap() + minPoint.getVector3fMap());

  // Final transform
  box.bboxQuaternion = eigenVectorsPCA;
  box.bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();
  box.cube_length = maxPoint.x - minPoint.x;
  box.cube_width = maxPoint.y - minPoint.y;
  box.cube_height = maxPoint.z - minPoint.z;

  return box;
}

template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to "+file << std::endl;
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size () << " data points from "+file << std::endl;

    return cloud;
}

template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}