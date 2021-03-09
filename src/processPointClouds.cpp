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


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // TODO:: Fill in the function to do voxel grid point reduction and region based filtering

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloud;

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

        pcl::PointXYZ point = cloud->points[index];
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

template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
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